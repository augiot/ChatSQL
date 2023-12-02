import os
import re
import copy
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import platform
from transformers import AutoTokenizer, AutoModel
# from utility.db_tools import Cur_db
# from utility.loggers import logger
from sentence_transformers import util
from prompt import table_schema, embedder,corpus_embeddings, corpus,In_context_prompt

MODEL_PATH = os.environ.get('MODEL_PATH', '/mnt/user2/workspace/Aug/model/chatglm3-6b')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).half().cuda()
model = model.eval() 
os_name = platform.system()

temp_table_prompt = """
\n表名:v_phm_floor_manage_result\n字段:stat_date:(日期),factory:(厂区),area_floor_id:(楼层),project:(机种),line_no:(线别),process:(制程),process_type:(制程类型),work_station:(工站),equipment_type:(机台类型),equipment_name:(机台名称),equipment_no:(机台SN),material_group:(产品类型),device_type:(设备类型),phase:(阶段),devicename:(设备名称),all_num:(总产量),ok_num:(ok产量),yield:(良率),oee_time:(时间稼动率),oee_pre:(性能稼动率),oee:(oee),ttl_time:(总时间),sum_normal_time:(各穴位总加工时长),normal_time:(加工时长),wait_time:(待料时长),alarm_time:(报警时长),offline_time:(未联机时长),discon_time:(通讯异常时长),adjust_time:(调机时长),warm_time:(暖机时长),ct_por:(标准ct),ct_avg:(实际平均ct),prod_corr:(一出N系数),cave_num:(穴位数),is_online:(是否开机),product:(产品)
"""
temp_history_prompt = """
问:本周的观澜的GL-B11-1F的238M的所有工站的设备综合效率\n答:"SELECT stat_date,factory,area_floor_id,project,work_station,设备综合效率 FROM  cnc_pdata.v_phm_floor_manage_result a WHERE stat_date='本周' and factory='GL' and area_floor_id='GL-B11-1F' and project='238M'  GROUP by is_bottleneck,stat_date,factory,area_floor_id,project,work_station"\n问:今日的观澜的GL-B05-4F的238M的CNC2上小U工站的所有线别的时间稼动率\n答:SELECT stat_date,factory,area_floor_id,project,work_station,line_no,时间稼动率 FROM  cnc_pdata.v_phm_floor_manage_result a WHERE stat_date='今日' and factory='GL' and area_floor_id='GL-B05-4F' and project='238M' and work_station='CNC2上小U'  GROUP by stat_date,factory,area_floor_id,project,work_station,line_no\n问:今天的观澜的GL-C06-5F的237M的CNC6工站的B的各机台的性能稼动率\n答:SELECT stat_date,factory,area_floor_id,project,work_station,line_no,equipment_name,性能稼动率 FROM  cnc_pdata.v_phm_floor_manage_result a WHERE stat_date='今天' and factory='GL' and area_floor_id='GL-C06-5F' and project='237M' and work_station='CNC6' and line_no='B'  GROUP by stat_date,factory,area_floor_id,project,work_station,line_no,equipment_name\n问:近一周的观澜的GL-B05-3F的233M的CNC2.5工站的B的各自机台的两小时OEE\nSELECT stat_date,factory,area_floor_id,project,work_station,line_no,equipment_name,两小时OEE FROM cnc_pdata.v_phm_floor_manage_result a WHERE stat_date='近一周' and factory='GL' and area_floor_id='GL-B05-3F' and project='233M' and work_station='CNC2.5' and line_no='B'  GROUP by hh_interval,stat_date,factory,area_floor_id,project,work_station,line_no,equipment_name"\n
"""

# chatbot_prompt = """
# 你是一个文本转SQL的生成器，你的主要目标是尽可能的协助用户，将输入的文本转换为正确的SQL语句。
# 上下文开始
# 表名和表字段来自以下表：
# """

query_template = """问: <user_input>
答: 
"""


def main():
    # db_con = Cur_db()
    # db_con.pymysql_cur()
    print("欢迎使用 Text2SQL 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    history = []
    while True:
        chatbot_prompt = """
你是一个文本转SQL的生成器，你的主要目标是尽可能的协助用户将输入的文本转换为正确的SQL语句。
上下文开始
生成的表名和表字段均来自以下表：
"""
        query = input("\n🧑用户：")
        if query == "stop":
            break
        if query == "clear":
            history = []
            command = 'cls' if os_name == 'Windows' else 'clear'
            os.system(command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        top_k = 3 
        query_embedding = embedder.encode(query, convert_to_tensor=True) # 与6张表的表名和输入的问题进行相似度计算
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0] 
        top_results = torch.topk(cos_scores, k=top_k) # 拿到topk=3的表名
        # 组合Prompt
        table_nums = 0 
        for score, idx in zip(top_results[0], top_results[1]):
            # 阈值过滤
            if score > 0.45:
                table_nums += 1
                chatbot_prompt += table_schema[corpus[idx]]
        if table_nums == 0:
            chatbot_prompt += temp_table_prompt
        chatbot_prompt += "上下文结束\n"
        # In-Context Learning
        if table_nums >= 2 and not history: # 如果表名大于等于2个，且没有历史记录，就加上In-Context Learning
            chatbot_prompt += In_context_prompt
        else:
            chatbot_prompt += temp_history_prompt
            
        #  加上查询模板
        chatbot_prompt += query_template
        # chatbot_prompt = chatbot_prompt.replace(" ", "")
        # 生成输入的prompt
        copy_query = copy.deepcopy(query)
        if history:
            query = query
        else:
            query = chatbot_prompt.replace("<user_input>", query)
        response, history = model.chat(tokenizer, query, history=history, temperature=0.1) # 生成SQL
        
        response = re.split("```|\n\n", response)
        print(response)
        for text in response:
            if "SELECT" in text:
                response = text
                break
        else:
            response = response[0]
        response = response.replace("\n", " ").replace("``", "").replace("`", "").strip()
        response = re.sub(' +',' ', response)
        print(f"🤖Text2SQL：{response}") 
        if "很抱歉" in response:
            continue
        # 结果查询
        # res = db_con.selectMany(response)
        # print("result table:", res)
        # query和sql入库
        # sql = "INSERT INTO query_sql_result (user_query, gen_sql) VALUES (%s, %s)"
        val = [copy_query, response]
        # db_con.insert(sql, val)
        # history = []

if __name__ == "__main__":
    main()