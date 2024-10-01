import psycopg2

import configure


def connect_to_pg():
    connection = psycopg2.connect(
        dbname=configure.dbname3,
        user=configure.user,
        password=configure.password,
        host=configure.host,
        port=configure.port
    )
    return connection

def get_tables_and_columns(conn):
    # 创建字典来存储列名和表名的映射
    table_mapping = {}

    # 创建游标对象
    cursor = conn.cursor()

    # 获取所有表名
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    tables = cursor.fetchall()

    # 遍历所有表，获取每个表的列
    for (table,) in tables:
        cursor.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = '{table}'
        """)
        columns = cursor.fetchall()

        # 将每个列名映射到它的表名
        for (column,) in columns:
            table_mapping[column] = table

    cursor.close()
    return table_mapping


# 连接到数据库
try:
    conn = connect_to_pg()
    # 调用函数并打印结果
    mapping = get_tables_and_columns(conn)
    print(mapping)

except Exception as e:
    print("Unable to connect to the database:", str(e))
finally:
    if conn:
        conn.close()
