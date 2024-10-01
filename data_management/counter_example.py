import re

import psycopg2
import json
import os
import configure
from enumerate_all_plans_by_join_order import generate_join_order_hints
from evaluate_cost_matrix import generate_hint_from_plan

time_regex = re.compile(r'Execution Time: ([\d\.]+) ms')

def connect_to_pg():
    connection = psycopg2.connect(
        dbname=configure.dbname,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection


def fetch_execution_plan2(connection, query):
    cursor = connection.cursor()

    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    cursor.execute(f"EXPLAIN (FORMAT JSON) {query}")
    plan = cursor.fetchone()
    cursor.close()
    return plan[0][0]

def fetch_execution_plan(connection, query, parameter):
    cursor = connection.cursor()

    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    query = cursor.mogrify(query, parameter).decode()

    cursor.execute(f"EXPLAIN (FORMAT JSON) {query}")
    plan = cursor.fetchone()
    cursor.close()
    return plan[0][0]

def fetch_actual_latency2(connection, query):
    cursor = connection.cursor()
    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    cursor.execute(query)
    explain_analyze_result = cursor.fetchall()
    latency = 0
    for row in explain_analyze_result:
        match = time_regex.search(row[0])
        if match:
            latency = float(match.group(1))
            break

    cursor.close()
    # latency = end_time - start_time
    return latency

def fetch_actual_latency(connection, query, parameters):
    cursor = connection.cursor()
    cursor.execute("SET max_parallel_workers_per_gather TO 0;")

    query = cursor.mogrify(query, parameters).decode()

    cursor.execute(query)
    explain_analyze_result = cursor.fetchall()
    latency = 0
    for row in explain_analyze_result:
        match = time_regex.search(row[0])
        if match:
            latency = float(match.group(1))
            break

    cursor.close()
    # latency = end_time - start_time
    return latency

def get_plans(connection, plan, Q1):
    hints = generate_join_order_hints(plan, 1000)
    plans = {}
    idx = 0
    for hint in hints:
        modified_plan_with_hint = fetch_execution_plan(connection, hint + " " + Q1)
        plans[f"plan {idx}"] = modified_plan_with_hint
        idx += 1

    with open("./counter_example_plans.json", 'w') as f:
        json.dump(plans, f, indent=4)

    return plans

#Q1 = "SELECT MIN(chn.name) AS uncredited_voiced_character, MIN(t.title) AS russian_movie FROM char_name AS chn, cast_info AS ci, company_name AS cn, company_type AS ct, movie_companies AS mc, role_type AS rt, title AS t WHERE ci.note LIKE '%(voice)%' AND ci.note LIKE '%(uncredited)%' AND cn.country_code = '[ru]' AND rt.role = 'actor' AND t.production_year > 2005 AND t.id = mc.movie_id AND t.id = ci.movie_id AND ci.movie_id = mc.movie_id AND chn.id = ci.person_role_id AND rt.id = ci.role_id AND cn.id = mc.company_id AND ct.id = mc.company_type_id;"
#Q2 = "SELECT MIN(chn.name) AS uncredited_voiced_character, MIN(t.title) AS russian_movie FROM char_name AS chn, cast_info AS ci, company_name AS cn, company_type AS ct, movie_companies AS mc, role_type AS rt, title AS t WHERE ci.note LIKE '%(voice)%' AND ci.note LIKE '%(uncredited)%' AND cn.country_code = '[ru]' AND rt.role = 'actor' AND t.production_year > 2004 AND t.id = mc.movie_id AND t.id = ci.movie_id AND ci.movie_id = mc.movie_id AND chn.id = ci.person_role_id AND rt.id = ci.role_id AND cn.id = mc.company_id AND ct.id = mc.company_type_id;"
#Q1 = "SELECT MIN(chn.name) AS voiced_char_name, MIN(n.name) AS voicing_actress_name, MIN(t.title) AS voiced_action_movie_jap_eng FROM aka_name AS an, char_name AS chn, cast_info AS ci, company_name AS cn, info_type AS it, keyword AS k, movie_companies AS mc, movie_info AS mi, movie_keyword AS mk, name AS n, role_type AS rt, title AS t WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code ='[us]' AND it.info = 'release dates' AND k.keyword IN ('hero','martial-arts','hand-to-hand-combat') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND n.gender ='f' AND n.name LIKE '%An%' AND rt.role ='actress' AND t.production_year > 2010 AND t.id = mi.movie_id AND t.id = mc.movie_id AND t.id = ci.movie_id AND t.id = mk.movie_id AND mc.movie_id = ci.movie_id AND mc.movie_id = mi.movie_id AND mc.movie_id = mk.movie_id AND mi.movie_id = ci.movie_id AND mi.movie_id = mk.movie_id AND ci.movie_id = mk.movie_id AND cn.id = mc.company_id AND it.id = mi.info_type_id AND n.id = ci.person_id AND rt.id = ci.role_id AND n.id = an.person_id AND ci.person_id = an.person_id AND chn.id = ci.person_role_id AND k.id = mk.keyword_id;"
#Q2 = "SELECT MIN(chn.name) AS voiced_char_name, MIN(n.name) AS voicing_actress_name, MIN(t.title) AS voiced_action_movie_jap_eng FROM aka_name AS an, char_name AS chn, cast_info AS ci, company_name AS cn, info_type AS it, keyword AS k, movie_companies AS mc, movie_info AS mi, movie_keyword AS mk, name AS n, role_type AS rt, title AS t WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code ='[us]' AND it.info = 'release dates' AND k.keyword IN ('hero','martial-arts','hand-to-hand-combat') AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND n.gender ='f' AND n.name LIKE '%An%' AND rt.role ='actress' AND t.production_year > 2015 AND t.id = mi.movie_id AND t.id = mc.movie_id AND t.id = ci.movie_id AND t.id = mk.movie_id AND mc.movie_id = ci.movie_id AND mc.movie_id = mi.movie_id AND mc.movie_id = mk.movie_id AND mi.movie_id = ci.movie_id AND mi.movie_id = mk.movie_id AND ci.movie_id = mk.movie_id AND cn.id = mc.company_id AND it.id = mi.info_type_id AND n.id = ci.person_id AND rt.id = ci.role_id AND n.id = an.person_id AND ci.person_id = an.person_id AND chn.id = ci.person_role_id AND k.id = mk.keyword_id;"

# json_path = f"../training_data/JOB/26a/parameter_new.json"
# with open(json_path, 'r') as file:
#     json_data = json.load(file)
#
# plan_path = f"../training_data/JOB/26a/all_plans_by_hybrid_new.json"
#
# with open(plan_path, 'r') as file2:
#     plans = json.load(file2).values()
#
#
# Q = "SELECT MIN(chn.name) AS character_name,\n       MIN(mi_idx.info) AS rating,\n       MIN(n.name) AS playing_actor,\n       MIN(t.title) AS complete_hero_movie\nFROM complete_cast AS cc,\n     comp_cast_type AS cct1,\n     comp_cast_type AS cct2,\n     char_name AS chn,\n     cast_info AS ci,\n     info_type AS it2,\n     keyword AS k,\n     kind_type AS kt,\n     movie_info_idx AS mi_idx,\n     movie_keyword AS mk,\n     name AS n,\n     title AS t\nWHERE cct1.kind = %s\n  AND cct2.kind LIKE %s\n  AND chn.name IS NOT NULL\n  AND (chn.name LIKE %s\n       OR chn.name LIKE %s)\n  AND it2.info = %s\n  AND k.keyword IN ('superhero',\n                    'marvel-comics',\n                    'based-on-comic',\n                    'tv-special',\n                    'fight',\n                    'violence',\n                    'magnet',\n                    'web',\n                    'claw',\n                    'laser')\n  AND kt.kind = %s\n  AND mi_idx.info > %s\n  AND t.production_year > %s\n  AND kt.id = t.kind_id\n  AND t.id = mk.movie_id\n  AND t.id = ci.movie_id\n  AND t.id = cc.movie_id\n  AND t.id = mi_idx.movie_id\n  AND mk.movie_id = ci.movie_id\n  AND mk.movie_id = cc.movie_id\n  AND mk.movie_id = mi_idx.movie_id\n  AND ci.movie_id = cc.movie_id\n  AND ci.movie_id = mi_idx.movie_id\n  AND cc.movie_id = mi_idx.movie_id\n  AND chn.id = ci.person_role_id\n  AND n.id = ci.person_id\n  AND k.id = mk.keyword_id\n  AND cct1.id = cc.subject_id\n  AND cct2.id = cc.status_id\n  AND it2.id = mi_idx.info_type_id;\n\n"

connection = connect_to_pg()
# parameter1 = json_data['parameter 941']
# parameter2 = json_data['parameter 941']
# parameter2[4] = '1920'

Q1 = "SELECT MIN(mi.info) AS movie_budget, MIN(mi_idx.info) AS movie_votes, MIN(n.name) AS writer, MIN(t.title) AS complete_violent_movie FROM complete_cast AS cc, comp_cast_type AS cct1, comp_cast_type AS cct2, cast_info AS ci, info_type AS it1, info_type AS it2, keyword AS k, movie_info AS mi, movie_info_idx AS mi_idx, movie_keyword AS mk, name AS n, title AS t WHERE cct1.kind  in ('cast', 'crew') AND cct2.kind  ='complete+verified' AND ci.note  in ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it1.info  = 'genres' AND it2.info  = 'votes' AND k.keyword  in ('murder', 'violence', 'blood', 'gore', 'death', 'female-nudity', 'hospital') AND mi.info  in ('Horror', 'Thriller') AND n.gender  = 'm' AND t.production_year  > 2000 AND t.id = mi.movie_id AND t.id = mi_idx.movie_id AND t.id = ci.movie_id AND t.id = mk.movie_id AND t.id = cc.movie_id AND ci.movie_id = mi.movie_id AND ci.movie_id = mi_idx.movie_id AND ci.movie_id = mk.movie_id AND ci.movie_id = cc.movie_id AND mi.movie_id = mi_idx.movie_id AND mi.movie_id = mk.movie_id AND mi.movie_id = cc.movie_id AND mi_idx.movie_id = mk.movie_id AND mi_idx.movie_id = cc.movie_id AND mk.movie_id = cc.movie_id AND n.id = ci.person_id AND it1.id = mi.info_type_id AND it2.id = mi_idx.info_type_id AND k.id = mk.keyword_id AND cct1.id = cc.subject_id AND cct2.id = cc.status_id;"
Q2 = "SELECT MIN(mi.info) AS movie_budget, MIN(mi_idx.info) AS movie_votes, MIN(n.name) AS writer, MIN(t.title) AS complete_violent_movie FROM complete_cast AS cc, comp_cast_type AS cct1, comp_cast_type AS cct2, cast_info AS ci, info_type AS it1, info_type AS it2, keyword AS k, movie_info AS mi, movie_info_idx AS mi_idx, movie_keyword AS mk, name AS n, title AS t WHERE cct1.kind  in ('cast', 'crew') AND cct2.kind  ='complete+verified' AND ci.note  in ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND it1.info  = 'genres' AND it2.info  = 'votes' AND k.keyword  in ('murder', 'violence', 'blood', 'gore', 'death', 'female-nudity', 'hospital') AND mi.info  in ('Horror', 'Thriller') AND n.gender  = 'm' AND t.production_year  > 2002 AND t.id = mi.movie_id AND t.id = mi_idx.movie_id AND t.id = ci.movie_id AND t.id = mk.movie_id AND t.id = cc.movie_id AND ci.movie_id = mi.movie_id AND ci.movie_id = mi_idx.movie_id AND ci.movie_id = mk.movie_id AND ci.movie_id = cc.movie_id AND mi.movie_id = mi_idx.movie_id AND mi.movie_id = mk.movie_id AND mi.movie_id = cc.movie_id AND mi_idx.movie_id = mk.movie_id AND mi_idx.movie_id = cc.movie_id AND mk.movie_id = cc.movie_id AND n.id = ci.person_id AND it1.id = mi.info_type_id AND it2.id = mi_idx.info_type_id AND k.id = mk.keyword_id AND cct1.id = cc.subject_id AND cct2.id = cc.status_id;"

plan1 = fetch_execution_plan2(connection, Q1)
plan2 = fetch_execution_plan2(connection, Q2)

connection.close()

plan_hint1 = generate_hint_from_plan(plan1)
#print(plan_hint1)
plan_hint2 = generate_hint_from_plan(plan2)
#print(plan_hint2)


# connection = connect_to_pg()
# query_with_hint11 = f"/*+ {plan_hint1}  */ EXPLAIN ANALYZE " + Q1
# cost11 = fetch_actual_latency2(connection, query_with_hint11)
# print(f'11 runing time: {cost11}')
# connection.close()
#
connection = connect_to_pg()
query_with_hint22 = f"/*+ {plan_hint2} */ EXPLAIN ANALYZE " + Q2
cost22 = fetch_actual_latency2(connection, query_with_hint22)
print(f'22 runing time: {cost22}')
connection.close()

# connection = connect_to_pg()
# query_with_hint21 = f"/*+ {plan_hint2} */ EXPLAIN ANALYZE " + Q1
# cost21 = fetch_actual_latency2(connection, query_with_hint21)
# print(f'21 runing time: {cost21}')
# connection.close()
#
connection = connect_to_pg()
query_with_hint12 = f"/*+ {plan_hint1} */ EXPLAIN ANALYZE " + Q2
cost12 = fetch_actual_latency2(connection, query_with_hint12)
print(f'12 runing time: {cost12}')
connection.close()

# query_with_hint21 = f"/*+ {plan_hint1} */ EXPLAIN ANALYZE " + Q
# cost21 = fetch_actual_latency(connection, query_with_hint11, parameter2)
# print(cost21)
#
# query_with_hint12 = f"/*+ {plan_hint2} */ EXPLAIN ANALYZE " + Q
# cost12 = fetch_actual_latency(connection, query_with_hint12, parameter1)
# print(cost12)
#
# query_with_hint22 = f"/*+ */ EXPLAIN ANALYZE " + Q
# cost22 = fetch_actual_latency(connection, query_with_hint22, parameter2)
# print(cost22)

# for p in plans:
#     plan_hint = generate_hint_from_plan(p)
#     #query_with_hint1 = f"/*+ {plan_hint} */ EXPLAIN ANALYZE " + Q
#     query = f"/*+  */ EXPLAIN ANALYZE " + Q
#     cost1 = fetch_actual_latency(connection, query, parameter2)
#     print(cost1)
