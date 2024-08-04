#!/usr/bin/env python
# coding: utf-8

# ### 01. 필요한 라이브러리 및 모듈 로드

# In[1]:


import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer
from encoder import PolyEncoder, CrossEncoder
from transform import SelectionJoinTransform, SelectionSequentialTransform, SelectionConcatTransform


# In[2]:


class Poly_function(object):
    def __init__(self, bert, model, tokenizer, device, context_transform, response_transform):
        self.bert = bert
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_transform = context_transform
        self.response_transform = response_transform
    
    def input_context(self, context):
        context_input_ids, context_input_masks = self.context_transform(context)
        contexts_token_ids_list_batch, contexts_input_masks_list_batch = [context_input_ids], [context_input_masks]
        long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch]
        contexts_token_ids_list_batch, contexts_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=self.device) for t in long_tensors)
        
        return contexts_token_ids_list_batch, contexts_input_masks_list_batch
    
    def response_input(self, candidates):
        responses_token_ids_list, responses_input_masks_list = self.response_transform(candidates)
        responses_token_ids_list_batch, responses_input_masks_list_batch = [responses_token_ids_list], [responses_input_masks_list]
        long_tensors = [responses_token_ids_list_batch, responses_input_masks_list_batch]
        responses_token_ids_list_batch, responses_input_masks_list_batch = (torch.tensor(t, dtype=torch.long, device=self.device) for t in long_tensors)
        
        return responses_token_ids_list_batch, responses_input_masks_list_batch
        
        
    def ctx_emb(self, contexts_token_ids_list_batch, contexts_input_masks_list_batch):
        with torch.no_grad():
            self.model.eval()
            
            ctx_out = self.model.bert(contexts_token_ids_list_batch, contexts_input_masks_list_batch)[0]  # [bs, length, dim]
            poly_code_ids = torch.arange(self.model.poly_m, dtype=torch.long).to(contexts_token_ids_list_batch.device)
            poly_code_ids = poly_code_ids.unsqueeze(0).expand(1, self.model.poly_m)
            poly_codes = self.model.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
            embs = self.model.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]
            
            return embs

        
    def cands_emb(self, responses_token_ids_list_batch, responses_input_masks_list_batch):
        with torch.no_grad():
            self.model.eval()
                
            batch_size, res_cnt, seq_length = responses_token_ids_list_batch.shape # res_cnt is 1 during training
            responses_token_ids_list_batch = responses_token_ids_list_batch.view(-1, seq_length)
            responses_input_masks_list_batch = responses_input_masks_list_batch.view(-1, seq_length)
            cand_emb = self.model.bert(responses_token_ids_list_batch, responses_input_masks_list_batch)[0][:,0,:] # [bs, dim]
            cand_emb = cand_emb.view(batch_size, res_cnt, -1) # [bs, res_cnt, dim]

            return cand_emb
        
    def score(self, embs, cand_emb):
        with torch.no_grad():
            self.model.eval()

            ctx_emb = self.model.dot_attention(cand_emb, embs, embs) # [bs, res_cnt, dim]
            dot_product = (ctx_emb*cand_emb).sum(-1)
        
            return dot_product
        

class Cross_function(object):
    def __init__(self, bert, model, tokenizer, device, concat_transform):
        self.bert = bert
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.concat_transform = concat_transform
    
    def input_text(self, context,response):
        text_input_ids, text_input_masks, text_segment_ids = self.concat_transform(context,response)
        text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = [text_input_ids], [text_input_masks], [text_segment_ids]
        long_tensors = [text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch]
        text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = (torch.tensor(t, dtype=torch.long, device=self.device) for t in long_tensors)
        
        return text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch
    
    def text_emb(self, text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch):
        batch_size, neg, dim = text_token_ids_list_batch.shape
        text_token_ids_list_batch = text_token_ids_list_batch.reshape(-1,dim)
        text_input_masks_list_batch = text_input_masks_list_batch.reshape(-1,dim)
        text_segment_ids_list_batch = text_segment_ids_list_batch.reshape(-1,dim)
        text_vec = self.model.bert(text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch)[0][:,0,:]
        
        model = self.model.linear
        model = model.cuda()
        score = model(text_vec)
        score = score.view(-1, 1)
        
        return score

class Filter_function(object):    
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
        
        def prediction(self, model, tokenizer, text):
            self.model.eval()
            tokenized_text = self.tokenizer(
                text,
                return_tensors = 'pt',
                truncation = True,
                add_special_tokens = True,
                max_length = 128
            )
            
            tokenized_text.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids = tokenized_text['input_ids'],
                    attention_mask = tokenized_text['attention_mask'],
                    token_type_ids = tokenized_text['token_type_ids']
                )
                
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                result = np.argmax(logits)
                
                return result