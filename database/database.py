import sqlite3

conn = sqlite3.connect('database/licence_plates.db')

c = conn.cursor()

#c.execute('CREATE TABLE vehicle_number_plate (id integer,plates text)')

#c.execute("INSERT INTO licence_plate VALUES (1,'CZ20FSE')")

# number_plate = [
#     (1,'DL8CAF5030'),
#     (2,'HR26DK8337'),
#     (3,'YX38590'),
#     (4,'KWI3MXR'),
#     (5,'COVID19'),
#     (6,'KA05NB1786'),
#     (7,'MH20DV2366'),
#     ]
# c.executemany("INSERT INTO vehicle_number_plate VALUES (?,?)", number_plate)

c.execute("SELECT * FROM vehicle_number_plate where id == 3 ")
print(c.fetchall())

conn.commit()

conn.close()