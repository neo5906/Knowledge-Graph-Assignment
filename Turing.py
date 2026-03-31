import spacy
import re
from pyvis.network import Network
from collections import defaultdict

# ==================== 1. 加载中文模型 ====================
try:
    nlp = spacy.load("zh_core_web_sm")
except OSError:
    print("未找到 zh_core_web_sm 模型，请执行：python -m spacy download zh_core_web_sm")
    exit(1)

# ==================== 2. 添加自定义概念实体（EntityRuler） ====================
# 在 spaCy 3.x 中，add_pipe 需要传入组件名称字符串，而不是组件实例
if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
else:
    ruler = nlp.get_pipe("entity_ruler")

concept_patterns = [
    {"label": "CONCEPT", "pattern": "图灵机"},
    {"label": "CONCEPT", "pattern": "图灵测试"},
    {"label": "CONCEPT", "pattern": "恩尼格玛密码机"},
    {"label": "CONCEPT", "pattern": "自动计算机"},
    {"label": "CONCEPT", "pattern": "计算机器与智能"},
]
ruler.add_patterns(concept_patterns)

# ==================== 3. 读取文本 ====================
with open("data/Turing_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

doc = nlp(text)

# ==================== 4. 实体抽取与消歧 ====================
# 消歧映射：将不同提及映射到标准名称
canonical_map = {
    "艾伦·图灵": "艾伦·图灵",
    "图灵": "艾伦·图灵",
    "他": "艾伦·图灵",          # 简单指代消解
    "剑桥大学": "剑桥大学",
    "普林斯顿大学": "普林斯顿大学",
    "阿隆佐·丘奇": "阿隆佐·丘奇",
    "布莱切利公园": "布莱切利公园",
    "国家物理实验室": "国家物理实验室",
    "图灵机": "图灵机",
    "图灵测试": "图灵测试",
    "恩尼格玛密码机": "恩尼格玛密码机",
    "自动计算机": "自动计算机",
    "ACE": "自动计算机",
    "计算机器与智能": "计算机器与智能",
    "伦敦": "伦敦",
    "二战": "二战",
    "1950年": "1950年",
    "1912年": "1912年",
    "1952年": "1952年",
    "2013年": "2013年",
}

# 存储实体（标准名 -> 信息）
entities = {}
for ent in doc.ents:
    mention = ent.text.strip()
    if len(mention) < 2:
        continue
    canonical = canonical_map.get(mention, mention)
    if canonical == "他":
        canonical = "艾伦·图灵"
    ent_type = ent.label_
    # 将自定义概念标记为 CONCEPT
    if ent.label_ == "CONCEPT" or "图灵" in canonical and ("机" in canonical or "测试" in canonical):
        ent_type = "CONCEPT"
    entities[canonical] = {
        "canonical": canonical,
        "type": ent_type,
        "mentions": [mention] if canonical not in entities else entities[canonical]["mentions"] + [mention]
    }

# 手动补充一些未识别的实体（可根据需要添加）
manual_entities = ["阿隆佐·丘奇", "自动计算机", "恩尼格玛密码机"]
for e in manual_entities:
    if e not in entities:
        entities[e] = {"canonical": e, "type": "PERSON" if e == "阿隆佐·丘奇" else "CONCEPT", "mentions": [e]}

unique_entities = list(entities.values())

# ==================== 5. 关系抽取 ====================
relations = []

# 5.1 基于依存句法的主谓宾模式
for sent in doc.sents:
    for token in sent:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subj = token.text
            verb = token.head.lemma_
            # 查找直接宾语
            obj = None
            for child in token.head.children:
                if child.dep_ == "dobj":
                    obj = child.text
                    break
            if obj:
                subj_canon = canonical_map.get(subj, subj)
                if subj_canon == "他":
                    subj_canon = "艾伦·图灵"
                obj_canon = canonical_map.get(obj, obj)
                if subj_canon in entities and obj_canon in entities:
                    relations.append((subj_canon, verb, obj_canon))
                else:
                    # 如果宾语是未记录的概念，动态加入实体
                    if "图灵" in obj and ("机" in obj or "测试" in obj):
                        entities[obj] = {"canonical": obj, "type": "CONCEPT", "mentions": [obj]}
                        relations.append((subj_canon, verb, obj))
            else:
                # 检查介词宾语
                for child in token.head.children:
                    if child.dep_ == "prep":
                        for grand in child.children:
                            if grand.dep_ == "pobj":
                                obj = grand.text
                                obj_canon = canonical_map.get(obj, obj)
                                if subj_canon in entities and obj_canon in entities:
                                    relations.append((subj_canon, verb + "_" + child.text, obj_canon))
                                break

# 5.2 基于正则表达式的特定关系模式
patterns = [
    (r"生于\s*([^。，]+)", "born_in"),
    (r"就读于\s*([^。，]+)", "studied_at"),
    (r"师从\s*([^。，]+)", "studied_under"),
    (r"工作于\s*([^。，]+)", "worked_at"),
    (r"设计了\s*([^。，]+)", "designed"),
    (r"提出了\s*([^。，]+)", "proposed"),
    (r"发表了\s*《?([^》]+)》?", "published"),
    (r"推动了\s*([^。，]+)", "promoted"),
]

for pattern, rel_type in patterns:
    for match in re.finditer(pattern, text):
        obj = match.group(1).strip()
        obj_canon = canonical_map.get(obj, obj)
        subj_canon = "艾伦·图灵"
        if subj_canon in entities and obj_canon in entities:
            relations.append((subj_canon, rel_type, obj_canon))
        else:
            # 若宾语未在实体中，添加到实体列表
            if obj_canon not in entities:
                entities[obj_canon] = {"canonical": obj_canon, "type": "CONCEPT", "mentions": [obj]}
                relations.append((subj_canon, rel_type, obj_canon))

# 去重
relations = list(set(relations))

# ==================== 6. 可视化 ====================
net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False)

# 添加节点
for ent in unique_entities:
    canonical = ent["canonical"]
    ent_type = ent["type"]
    color = {
        "PERSON": "#FFA07A",
        "ORG": "#87CEFA",
        "GPE": "#98FB98",
        "CONCEPT": "#DDA0DD",
        "DATE": "#F0E68C",
        "WORK_OF_ART": "#FFDAB9"
    }.get(ent_type, "#CCCCCC")
    net.add_node(canonical, label=canonical, title=f"类型: {ent_type}", color=color)

# 添加边
for subj, rel, obj in relations:
    if subj not in net.get_nodes():
        net.add_node(subj, label=subj, color="#CCCCCC")
    if obj not in net.get_nodes():
        net.add_node(obj, label=obj, color="#CCCCCC")
    net.add_edge(subj, obj, title=rel, label=rel, arrows="to", font={"size": 12})

# 设置物理引擎
net.set_options("""
var options = {
  "physics": {
    "enabled": true,
    "stabilization": {"iterations": 200},
    "repulsion": {
      "nodeDistance": 200,
      "centralGravity": 0.2
    }
  }
}
""")

net.save_graph("output/turing_kg_zh.html")
print("知识图谱已生成，请打开 output/turing_kg_zh.html 查看。")