#!/usr/bin/env python
# coding: utf-8

# ### 입력 발화문과 답변 후보군을 각각 BERT의 input 형태에 맞게 변경해주는 과정
# - SelectSequentialTransform(object): candidate에 대한 변경
# - SelectionJoinTransform(object): context에 대한 변경
# - SelectionConcatTransform(object): cross-encoder input에 대한 변경(context, candidate 함께 받음)

# In[2]:


class SelectionSequentialTransform(object):
    """ candidate tokenizer """
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __call__(self, texts):
        input_ids_list, input_masks_list = [],[]
        for text in texts:
            tokenized_dict = self.tokenizer.encode_plus(text, padding="max_length", max_length=40, truncation=True)
            input_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['attention_mask']
            input_ids_list.append(input_ids)
            input_masks_list.append(input_masks)
        
        return input_ids_list, input_masks_list


# In[1]:


class SelectionJoinTransform(object):
    """Poly Encoder"""
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.tokenizer.add_tokens(['\n'], special_tokens=False)
        self.pad_id = 0
        
    def __call__(self, texts):
        context = '[SEP]'.join(texts)
        tokenized_dict = self.tokenizer.encode_plus(context)
        input_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['attention_mask']
        input_ids = input_ids[-self.max_len:]
        input_ids[0] = self.cls_id
        input_masks = input_masks[-self.max_len:]
        input_ids += [self.pad_id] * (self.max_len - len(input_ids))
        input_masks += [0] * (self.max_len - len(input_masks))
        assert len(input_ids) == self.max_len
        assert len(input_masks) == self.max_len
        return input_ids, input_masks


class SelectionConcatTransform(object):
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cls_id = self.tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        self.tokenizer.add_tokens(['\n'], special_tokens=False)
        self.pad_id = 0

    def __call__(self, context, responses):
        context = '[SEP]'.join(context)
        tokenized_dict = self.tokenizer.encode_plus(context)
        context_ids, context_masks, context_segment_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']
        ret_input_ids = []
        ret_input_masks = []
        ret_segment_ids = []
        for response in responses:
            tokenized_dict = self.tokenizer.encode_plus(response)
            response_ids, response_masks, response_segment_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']
            response_segment_ids = [1]*(len(response_segment_ids)-1)
            input_ids = context_ids + response_ids[1:]
            input_ids = input_ids[-self.max_len:]
            input_masks = context_masks + response_masks[1:]
            input_masks = input_masks[-self.max_len:]
            input_segment_ids = context_segment_ids + response_segment_ids
            input_segment_ids = input_segment_ids[-self.max_len:]
            input_ids[0] = self.cls_id
            input_ids += [self.pad_id] * (self.max_len - len(input_ids))
            input_masks += [0] * (self.max_len - len(input_masks))
            input_segment_ids += [0] * (self.max_len - len(input_segment_ids))
            assert len(input_ids) == self.max_len
            assert len(input_masks) == self.max_len
            assert len(input_segment_ids) == self.max_len
            ret_input_ids.append(input_ids)
            ret_input_masks.append(input_masks)
            ret_segment_ids.append(input_segment_ids)
        return ret_input_ids, ret_input_masks, ret_segment_ids