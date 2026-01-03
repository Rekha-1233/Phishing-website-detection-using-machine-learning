import sqlite3

conn = sqlite3.connect("database.db")  # or whatever your DB file is named
cur = conn.cursor()

print("🎯 USERS TABLE SCHEMA:")
for row in cur.execute("PRAGMA table_info(users)"):
    print(row)

print("\n📊 PREDICTIONS TABLE SCHEMA:")
for row in cur.execute("PRAGMA table_info(predictions)"):
    print(row)

conn.close()
