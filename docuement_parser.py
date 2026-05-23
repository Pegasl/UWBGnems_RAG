import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import os

def clean_and_extract_doc(json_filepath):
    """
    清洗 JSON 文档：
    1. 剔除页眉、页脚等噪声
    2. 剔除没有 Caption 的无效图片
    3. 截断“致谢”、“参考文献”等正文无关的尾部内容
    """
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 定义需要直接丢弃的元素类型
    NOISE_TYPES = {'header', 'footer', 'page_number', 'aside_text'}
    
    # 2. 定义触发截断的“停止标题”关键词 (全部小写，方便匹配)
    STOP_SECTION_TITLES = {
        "acknowledgements", 
        "acknowledgments", 
        "references", 
        "notes and references", 
        "author contributions", 
        "conflicts of interest",
        "declaration of competing interest",
        "data availability"
    }
    
    full_markdown_lines = []
    image_records = [] 
    
    current_page = -1
    skip_remaining = False  # 截断开关

    # 3. 遍历数据
    for item in data:
        # 如果截断开关已被触发，直接跳过后续所有内容
        if skip_remaining:
            continue

        item_type = item.get("type")
        page_idx = item.get("page_idx", 0)
        
        # 翻页标记
        if page_idx != current_page:
            full_markdown_lines.append(f"\n<!-- Page {page_idx} -->\n")
            current_page = page_idx

        # 过滤排版噪声
        if item_type in NOISE_TYPES:
            continue
            
        # 处理文本与标题
        elif item_type == "text":
            text_content = item.get("text", "").strip()
            if not text_content:
                continue
                
            text_level = item.get("text_level")
            
            # 【核心逻辑】：检测是否是一级标题，且标题内容属于“停止词”
            if text_level == 1:
                # 转换为小写并去除首尾空格以进行精确匹配
                clean_title = text_content.lower()
                if clean_title in STOP_SECTION_TITLES:
                    print(f"检测到尾部区块 [{text_content}]，已截断后续所有内容 (Page {page_idx})。")
                    skip_remaining = True
                    continue
                
                # 如果是正常的正文标题，转为 Markdown
                md_title = f"{'#' * text_level} {text_content}"
                full_markdown_lines.append(md_title)
            else:
                # 普通段落文本
                full_markdown_lines.append(text_content)

        # 处理列表 (如 Reference 里的条目也会在这里，但开关触发后就不会走到这了)
        elif item_type == "list":
            list_items = item.get("list_items", [])
            for li in list_items:
                full_markdown_lines.append(f"- {li.strip()}")

        # 处理图片，过滤无 Caption 的无效图
        elif item_type == "image":
            captions = item.get("image_caption", [])
            
            if not captions:
                continue
                
            caption_text = " ".join(captions).strip()
            if not caption_text:
                continue
                
            img_path = item.get("img_path", "")
            full_markdown_lines.append(f"\n![{caption_text}]({img_path})\n")
            
            image_records.append({
                "page_idx": page_idx,
                "img_path": img_path,
                "caption": caption_text
            })

    full_text = "\n\n".join(full_markdown_lines)
    return full_text, image_records


def chunk_document(full_text, chunk_size=600, chunk_overlap=100):
    """
    使用 RecursiveCharacterTextSplitter 对长文本进行分块
    """
    # 这个分块器会优先按段落(\n\n)切分，如果段落太长再按单行(\n)或句号切分，最大程度保留语义完整性
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       # 每个 Chunk 的最大字符数 (根据你的 Embedding 模型限制调整，通常 500-1000)
        chunk_overlap=chunk_overlap, # 相邻 Chunk 之间的重叠字符数，防止一句话被从中间截断
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
    )
    
    chunks = text_splitter.split_text(full_text)
    return [c.strip() for c in chunks if c.strip()]