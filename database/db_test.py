import sqlite3

conn = sqlite3.connect('test.db')

c = conn.cursor()

# c.execute(""" CREATE TABLE customer (
#     Name text,
#     id integer,
#     email text
# )
# """)

many_customers = [
    ('bandi',4,'bandi@gmail.com'),
    ('pathi',5,'pathi@gmail.com'),
    ('rohit',6,'rohit@gmail.com'),
    ]


c.executemany(" INSERT INTO customer VALUES (?,?,?)",many_customers)


conn.commit()

conn.close()