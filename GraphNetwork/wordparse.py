
import streamlit as st
from docx import Document

def extract_text_from_docx(docx_file_path):
    # 打开Word文档
    doc = Document(docx_file_path)
    # 初始化标题和正文
    title = ''
    content = ''
    # 初始化标题等级和编号
    level = 0
    number = []
    # 遍历Word文档中的每一个段落
    for para in doc.paragraphs:
        # 如果这个段落是标题，则将标题更新为这个段落的文本
        if para.style.name.startswith("Heading"):
            # 判断标题的等级
            new_level = int(para.style.name[-1])
            # 如果新的标题等级比当前标题等级高，则在编号末尾添加0
            while new_level > level:
                number.append(0)
                level += 1
            # 如果新的标题等级比当前标题等级低，则删除编号中比新等级高的数字
            while new_level < level:
                number.pop()
                level -= 1
            # 在编号末尾添加新的数字
            number[-1] += 1
            # 根据标题的级别和编号，生成标题前面的编号字符串
            if level == 1:
                prefix = str(number[0])
            elif level == 2:
                prefix = f"{number[0]}.{number[1]}"
            else:
                prefix = '.'.join(str(n) for n in number)
            # 将编号和标题合并
            title = f"{'#' * level} {prefix} {para.text}"
            # 如果是第一个大标题，则在标题前面加上1
            if level == 1 and number == [1]:
                title = "# 1 " + para.text
            # 如果是第二个大标题，则在标题前面加上2
            if level == 1 and number == [2]:
                title = "# 2 " + para.text
            # 将标题添加到正文中
            content += title + '\n'
        # 如果这个段落不是标题，则将这个段落的文本添加到正文中
        else:
            content += para.text.strip() + '\n'
    return content

st.title("上传并解析 docx 文件")

# 显示上传文件组件
uploaded_file = st.file_uploader("上传文件", type="docx")

if uploaded_file is not None:
    # 提取文档中的正文
    content = extract_text_from_docx(uploaded_file)
    # 显示正文
    st.markdown(content)