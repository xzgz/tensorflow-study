import json

ord_map = '/media/xzgz/Ubuntu/Ubuntu/Caffe/eclipse-caffe/crnn/crnn-tensorflow/data/char_dict/ord_map.json'
ord_map_t = '/media/xzgz/Ubuntu/Ubuntu/Caffe/eclipse-caffe/crnn/crnn-tensorflow/data/char_dict/ord_map_'
char_dict = '/media/xzgz/Ubuntu/Ubuntu/Caffe/eclipse-caffe/crnn/crnn-tensorflow/data/char_dict/char_dict.json'
char_dict_t = '/media/xzgz/Ubuntu/Ubuntu/Caffe/eclipse-caffe/crnn/crnn-tensorflow/data/char_dict/char_dict_'


with open(char_dict, 'r', encoding='utf-8') as json_f:
    res = json.load(json_f)
print(res)


# with open(ord_map) as f:
#     data = json.load(f)
# 
# word_unicode_dict = {}
# ranked_wordkey_list = sorted(data.keys(), key = lambda wordkey: int(wordkey))
# print(ranked_wordkey_list)
# for key in ranked_wordkey_list:
#     word_unicode_dict[key] = data[key]
# print(word_unicode_dict)
# with open(ord_map_t+'t.json', 'w') as f:
#     json.dump(word_unicode_dict, f)

# word_unicode_dict = data
# print(word_unicode_dict['0'])
# word_unicode_dict['0'] = 3
# print(word_unicode_dict['0'])
# print(data['0'])


test_anno_path = '/media/xzgz/Ubuntu/Ubuntu/ICPR/train_test_image3/Test/sample.txt'
train_anno_path = '/media/xzgz/Ubuntu/Ubuntu/ICPR/train_test_image3/Train/sample.txt'
word_label = {}
label_count = {}
label_index = 1

with open(test_anno_path, 'r') as anno_file:
    info = [tmp.strip().split() for tmp in anno_file.readlines()]
    labels = []
    imagenames = []
    for ml in info:
        if len(ml) == 1:
            print(ml[0] + ' have no label!')
            continue
        label_index += 1
         
        for l in ml[1]:
            temp = ord(l)
            if temp not in label_count:
                label_count[temp] = 0
                word_label[temp] = l
            label_count[temp] += 1
 
with open(train_anno_path, 'r') as anno_file:
    info = [tmp.strip().split() for tmp in anno_file.readlines()]
    labels = []
    imagenames = []
    for ml in info:
        if len(ml) == 1:
            print(ml[0] + ' have no label!')
            continue
        label_index += 1
         
        for l in ml[1]:
            temp = ord(l)
            if temp not in label_count:
                label_count[temp] = 0
                word_label[temp] = l
            label_count[temp] += 1

def morethan_freqi_label(ranked_key_list, i):
    label_height_freq = []
    for key in ranked_key_list:
        if label_count[key] >= i:
            label_height_freq.append(key)
        else:
            return label_height_freq
         
    return label_height_freq


ranked_key_list = sorted(label_count.keys(), key = lambda l: (-label_count[l]))
total_label_cnt = len(label_count)
print('total_label_cnt = ', total_label_cnt)

min_cnt = 1
label_height_freq = morethan_freqi_label(ranked_key_list, min_cnt)
min_key, max_key = min(label_height_freq), max(label_height_freq)
label_height_freq_cnt = len(label_height_freq)
print('max = ', max_key, 'min = ', min_key)
print('label_height_freq_cnt = ', label_height_freq_cnt)
word_unicode_dict = {}
unicode_hex_dict = {}
index = 0
for key in label_height_freq:
    word_unicode_dict[index] = str(key)
    unicode_hex_dict[key] = word_label[key]
    index += 1
print(word_label[max_key])
print(unicode_hex_dict[max_key])
# s = chr(max_key + 1)
# print('The char out of dict is labeled as: ', s)
# word_unicode_dict[index] = str(max_key + 1)
# unicode_hex_dict[max_key + 1] = s

# print(word_unicode_dict)
# print(unicode_hex_dict)
# ord_map_t += str(max_key+1) + '_' + str(label_height_freq_cnt+1) + '_' + str(min_cnt) + '.json'
# char_dict_t += str(max_key+1) + '_' + str(label_height_freq_cnt+1) + '_' + str(min_cnt) + '.json'
ord_map_t += str(max_key) + '_' + str(label_height_freq_cnt) + '_' + str(min_cnt) + '.json'
char_dict_t += str(max_key) + '_' + str(label_height_freq_cnt) + '_' + str(min_cnt) + '.json'
with open(ord_map_t, 'w') as f:
    json.dump(word_unicode_dict, f)
with open(char_dict_t, 'w', encoding='utf-8') as f:
    json.dump(unicode_hex_dict, f)


