# -*- coding: utf-8 -*-

import sys
sys.path.append("./")
sys.path.append("../")

# jieba分词使用示例
import jieba
import pandas as pd

# ============================================================================
# 读取停用词
# ============================================================================
# 停用词是指在文本处理中需要过滤掉的词，如"的"、"了"、"在"等无实际意义的词
# 停用词文件每行一个词，不需要分隔符
# 使用pd.read_csv读取停用词文件
# header=None: 文件没有表头
# names=['stopword']: 指定列名为'stopword'
# encoding='utf-8': 指定文件编码
# quoting=3: 使用QUOTE_NONE，不处理引号，避免解析错误
# on_bad_lines='skip': 跳过问题行
stopwords_df = pd.read_csv('data/text_data/stopwords.txt', 
                           header=None, 
                           names=['stopword'], 
                           encoding='utf-8',
                           quoting=3,
                           on_bad_lines='skip')
# 将DataFrame中的stopword列转换为Python列表
stopwords = stopwords_df['stopword'].tolist()
# 去除每个词的空白字符并过滤空值

# 对停用词列表进行预处理：去除每个词的首尾空白字符，并过滤掉空值和None值
# word.strip() 作用：去除字符串首尾的空白字符（包括空格、制表符、换行符等）
# pd.notna(word) 作用：检查词是否不是NaN或None
# word.strip() 作用：确保去除空白后的词不为空字符串
stopwords = [word.strip() for word in stopwords if pd.notna(word) and word.strip()]


# ============================================================================
# 定义分词函数
# ============================================================================
def cut_words(line, words_min=2):
    """
    对文本进行分词，并过滤停用词和短词
    
    参数说明:
    -----------
    line : str
        待分词的文本字符串
    words_min : int, default=2
        最小词长度，长度小于此值的词将被过滤
    
    返回值:
    -------
    list
        分词后的词语列表
    
    处理步骤:
    --------
    1. 使用jieba进行分词
    2. 过滤掉长度小于words_min的短词
    3. 过滤掉停用词
    """
    # 使用jieba.lcut进行分词，返回分词结果列表
    line_segments = jieba.lcut(line)
    
    # 过滤短词：保留长度大于等于words_min的词
    # lambda x: len(x) >= words_min 是一个匿名函数，判断词长度是否满足要求
    line_segments = filter(lambda x: len(x) >= words_min, line_segments)
    
    # 过滤停用词：保留不在停用词表中的词
    # lambda x: x not in stopwords 判断词是否为停用词
    line_segments = filter(lambda x: x not in stopwords, line_segments)
    
    # 将过滤器对象转换为列表返回
    return list(line_segments)


# ============================================================================
# 示例使用
# ============================================================================
# 定义测试文本
# 这是一段关于资金审批流程的业务文本
text_demo = "通过资料审核与电话沟通将被客户审批通过借款金额10000元操作人小明审批的间2020年10月5日经过电话核查客户所有资料均为本人提供交流响应2020年9月28日顺利完成"

# 对文本进行分词
segs = cut_words(text_demo)

# 打印分词结果
print("分词结果:")
print(segs)
