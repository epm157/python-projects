import bcrypt

passwd = b'password'
entered = b'password'

salt1 = bcrypt.gensalt()
hashed1 = bcrypt.hashpw(passwd, salt1)

salt2 = bcrypt.gensalt()
hashed2 = bcrypt.hashpw(entered, salt2)


if bcrypt.checkpw(entered, hashed1):
    print("match")
    print(hashed1)
    print(hashed2)
else:
    print("does not match")