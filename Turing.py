import spacy
import re
from pyvis.network import Network
from collections import defaultdict
from spacy.pipeline import EntityRuler

with open("data/Turing_Wiki.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 加载spaCy中文模型
nlp = spacy.load("zh_core_web_sm")
print("已加载模型 zh_core_web_sm")


# 添加自定义实体规则
if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
else:
    ruler = nlp.get_pipe("entity_ruler")

# 根据维基百科内容，总结出常见的概念实体、人名、机构等

# 核心概念实体（CONCEPT）
concept_patterns = [
    # 理论相关
    {"label": "CONCEPT", "pattern": "图灵机"},
    {"label": "CONCEPT", "pattern": "图灵测试"},
    {"label": "CONCEPT", "pattern": "可计算性理论"},
    {"label": "CONCEPT", "pattern": "计算理论"},
    {"label": "CONCEPT", "pattern": "停机问题"},
    {"label": "CONCEPT", "pattern": "丘奇-图灵论题"},
    {"label": "CONCEPT", "pattern": "人工智能"},
    {"label": "CONCEPT", "pattern": "机器智能"},
    # 密码破译相关
    {"label": "CONCEPT", "pattern": "恩尼格玛密码机"},
    {"label": "CONCEPT", "pattern": "Enigma"},
    {"label": "CONCEPT", "pattern": "布莱切利园"},
    {"label": "CONCEPT", "pattern": "密码分析"},
    # 计算机设计
    {"label": "CONCEPT", "pattern": "自动计算机"},
    {"label": "CONCEPT", "pattern": "ACE"},
    {"label": "CONCEPT", "pattern": "曼彻斯特大学"},
    {"label": "CONCEPT", "pattern": "曼彻斯特马克1号"},
    # 其他重要概念
    {"label": "CONCEPT", "pattern": "形态发生"},
    {"label": "CONCEPT", "pattern": "化学阉割"},
    {"label": "CONCEPT", "pattern": "皇家赦免"},
]

# 人物实体（PERSON）
person_patterns = [
    {"label": "PERSON", "pattern": "艾伦·图灵"},
    {"label": "PERSON", "pattern": "图灵"},
    {"label": "PERSON", "pattern": "阿隆佐·丘奇"},
    {"label": "PERSON", "pattern": "约翰·冯·诺伊曼"},
    {"label": "PERSON", "pattern": "冯·诺伊曼"},
    {"label": "PERSON", "pattern": "麦克斯·纽曼"},
    {"label": "PERSON", "pattern": "琼·克拉克"},
    {"label": "PERSON", "pattern": "克里斯托弗·莫科姆"},
    {"label": "PERSON", "pattern": "安德鲁·霍奇斯"},
]

# 组织机构（ORG）
org_patterns = [
    {"label": "ORG", "pattern": "剑桥大学"},
    {"label": "ORG", "pattern": "国王学院"},
    {"label": "ORG", "pattern": "普林斯顿高等研究院"},
    {"label": "ORG", "pattern": "布莱切利公园"},
    {"label": "ORG", "pattern": "国家物理实验室"},
    {"label": "ORG", "pattern": "英国皇家学会"},
]

# 地理位置（GPE/LOC）
geo_patterns = [
    {"label": "GPE", "pattern": "伦敦"},
    {"label": "GPE", "pattern": "英国"},
    {"label": "GPE", "pattern": "英格兰"},
    {"label": "GPE", "pattern": "剑桥"},
    {"label": "GPE", "pattern": "普林斯顿"},
    {"label": "GPE", "pattern": "曼彻斯特"},
]

# 日期（DATE）
date_patterns = [
    {"label": "DATE", "pattern": "1912年"},
    {"label": "DATE", "pattern": "1954年"},
    {"label": "DATE", "pattern": "1952年"},
    {"label": "DATE", "pattern": "2009年"},
    {"label": "DATE", "pattern": "2013年"},
    {"label": "DATE", "pattern": "第二次世界大战"},
    {"label": "DATE", "pattern": "二战"},
]

# 合并所有模式
all_patterns = (concept_patterns + person_patterns + org_patterns +
                geo_patterns + date_patterns)
ruler.add_patterns(all_patterns)
print(f"已添加 {len(all_patterns)} 个自定义实体规则")

# ==================== 4. 实体抽取 ====================
print("\n正在进行命名实体识别...")
doc = nlp(text)

# 存储实体信息
entities_list = []           # 所有实体提及
entity_counts = defaultdict(int)     # 提及次数
entity_mentions = defaultdict(list)  # 提及列表

for ent in doc.ents:
    mention = ent.text.strip()
    if len(mention) < 2:
        continue

    # 获取实体类型
    ent_type = ent.label_
    # 将自定义概念统一为CONCEPT
    if ent.label_ == "CONCEPT":
        ent_type = "CONCEPT"

    entities_list.append({
        "mention": mention,
        "type": ent_type,
        "start": ent.start_char,
        "end": ent.end_char
    })
    entity_counts[mention] += 1
    entity_mentions[mention].append(mention)

# 去重后的唯一实体
unique_entities = []
seen = set()
for mention in entity_counts:
    if mention not in seen:
        seen.add(mention)
        # 确定类型：取第一次出现的类型（同一提及可能被识别为不同类型，取第一个）
        typ = next((e["type"] for e in entities_list if e["mention"] == mention), "UNKNOWN")
        unique_entities.append({
            "canonical": mention,
            "type": typ,
            "mentions": list(set(entity_mentions[mention])),  # 去重
            "count": entity_counts[mention]
        })

# 按出现频率排序并打印
unique_entities_sorted = sorted(unique_entities, key=lambda x: x["count"], reverse=True)
print(f"\n共识别出 {len(entities_list)} 个实体提及，去重后 {len(unique_entities)} 个唯一实体")
print("\n实体列表（按出现频率排序）:")
for ent in unique_entities_sorted:
    print(f"  {ent['canonical']} ({ent['type']}) - 出现 {ent['count']} 次")

# 可视化知识图谱（仅实体节点）
print("\n正在生成知识图谱可视化...")

# 创建pyvis网络
net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False)

# 节点颜色映射
color_map = {
    "PERSON": "#FFA07A",      # 人物
    "ORG": "#87CEFA",         # 组织
    "GPE": "#98FB98",         # 地理政治实体
    "LOC": "#90EE90",         # 地点
    "CONCEPT": "#DDA0DD",     # 概念
    "DATE": "#F0E68C",        # 日期
    "NORP": "#B0C4DE",        # 民族/宗教/政治团体
    "PRODUCT": "#FFB6C1",     # 产品
    "WORK_OF_ART": "#FFDAB9", # 作品
    "CARDINAL": "#CD853F",    # 数字
    "UNKNOWN": "#CCCCCC",     # 未知
}

# 添加节点
for ent in unique_entities_sorted:
    canonical = ent["canonical"]
    ent_type = ent["type"]
    color = color_map.get(ent_type, "#CCCCCC")
    # 悬停信息：显示类型、出现次数、提及（限5个）
    mentions_show = '、'.join(ent["mentions"][:5])
    if len(ent["mentions"]) > 5:
        mentions_show += f" 等{len(ent['mentions'])}种提及"
    title = f"类型: {ent_type}<br>出现次数: {ent['count']}<br>提及: {mentions_show}"
    net.add_node(canonical, label=canonical, title=title, color=color)

# 设置图形选项
net.set_options("""
var options = {
  "nodes": {
    "font": {"size": 14, "face": "SimHei"},
    "borderWidth": 1,
    "shadow": true
  },
  "edges": {
    "arrows": {"to": {"enabled": false}},
    "color": {"color": "#848484", "highlight": "#848484"}
  },
  "physics": {
    "enabled": true,
    "stabilization": {"iterations": 200},
    "repulsion": {
      "nodeDistance": 200,
      "centralGravity": 0.2,
      "springLength": 200
    }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 200
  }
}
""")

# 保存HTML文件
import os
if not os.path.exists("output"):
    os.makedirs("output")
output_file = "output/turing_entities_wiki.html"
net.save_graph(output_file)
print(f"知识图谱已生成，请打开 {output_file} 查看")
print(f"共包含 {len(unique_entities)} 个实体节点")