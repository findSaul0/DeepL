#variabili in python
name= "Antonio"
age=21
code = 10.1
new_patient= True
print(name,age,new_patient,code)

#input in python
name = input("Qual è il tuo nome?")
print("Ciao",name)

#conversione tipo
date = "1999"
age = 2021 - int(date)
print(age)

#metodi per convertire nel tipo indicato
int()
float()
bool()
str()

#somma tra due numeri
first = float(input("Insert firt number:"))
second = float(input("Inser second number:"))

#attenzione all OPERATORE + perche viene utilizzato per concatenare
sum = first + second
print("Sum:" + str(sum))
print("Ci"+"ao")

#string
course = 'Python for Beginners'
print("Numero dove si trova la B:"+ str(course.find('B')))
print(course.replace('for','ciao'))
print('Python' in course)

#operazioni
print(10+3)
print(10-3)
print(10 // 3)
print(10 / 3)
print(10 ** 3)

x = 10
x +=3
print(x)

#operatori logici
#and
#or
#not

#if
temperature = 3
if temperature >20:
    print("Fa caldo")
elif temperature>30:
    print("Fa troppo caldo")
else:
    print("Fa freddo")

# WHILE
l = 1
while l <= 5:
    print(l)
    #print(l * '*') in questo caso avremo un intero moltiplicato per una stringa
    l = l + 1

# LISTE
print('SEZIONE LISTE')
names = list(["Antonio","Giuseppe","Marco"])
surname = list(["Ciao", "Ciao", "Ciao"])
ciao = "CIAO"
print(list(ciao))
total = names + surname
print("TOTAL = " , total)
names[1]= "Andrea"
print(names)
print(names[0]) #stampiamo il primo elemento della lista
print(names[-1])#stampiamo l'ultimo elemento in quanto è possibile considerare la lista come una struttura circolare
print(names[0:1])#stampiamo i nomi dalla posizione 0 alla posizione 3

numbers = list([1,2,3,4,5])
numbers.append(6)#aggiungiamo alla fine della lista il numero inserito
numbers.insert(0,8)#aggiungiamo il numero 8 alla posizione 0 della lista
numbers.remove(3)#rimoviamo il numero 3 dalla lista
numbers.pop()#rimuovi l'ultimo elemento della lista
numbers.pop(2)#rimuove l'elemento in posizione 2 nella lista
#numbers.clear()#rimuove tutti gli elementi della lista
print(1 in numbers)#ritorna un valore booleano per indicare se nella lista vi è un numero 1
print(len(numbers))#stampa la lunghezza della lista
print(numbers)
# PER QUANTO RIGUARA LA FUNZIONE SORT POSSIAMO PASSARE A QUESTA UNA KEY ED UN VALORE REVERSE
# LA KEY E UNA FUNZIONE, AD ESEMPIO LEN, DOVE IN QUESTO CASO ORDINA GLI ELEMENTI DELLA LISTA IN BASE ALLA LORO LUNGHEZZA
# REVERSE ASSUME UN VALORE BOOL CHE CONSENTIRA DI EFFETTUARE IL REVERSE DELLA LISTA

print("FOR")
# FOR
for o in numbers:
    print(o)

print("WHILE")
# IL FOR SOPRA EQUIVALE AL WHILE SOTTOSTANTE
j = 0
while j < len(numbers):
    print(numbers[j])
    j = j + 1

print("FUNZIONE RANGE")
# FUNZIONE RANGE
for o in range(5): #for da 0 a 4
    print(o)

print("FUNZIONE CREATA")
# FUNZIONE CREATA DA NOI
def print_tre_volte():
    print("ciao 1")
    print("ciao 2")
    print("ciao 3")

print_tre_volte()

print("FUNZIONE SOMMA")
def sommatrice(x,y):
    c = x + y
    #print("La somma dei due numeri e:",str(c))
    return (c)

print(sommatrice(10,5))

# POSSIAMO UTILIZZARE UNA VARIABILE GLOBALE ANCHE IN UNA FUNZIONE
x = 15
def stampa_variabile():
    global x
    x+=2
    return (x)

print(stampa_variabile())

# TRY,EXCEPT, AND FINNALY
def divisore(a,b):
    try:
        risultato = a / b
        print("Il risultato della divisione è :" + str(risultato))
    except ZeroDivisionError:
        print("Il numero è stato diviso per zero")
    finally:
        print("Grazie")

primo_fattore = input("Inserisci il primo numero:")
secondo_fattore = input("Inserisci il secondo numero:")

divisore(int(primo_fattore),int(secondo_fattore))

# DIZIONARI {chiave:valore}
# POSSIAMO RICHIAMARE I VALORI ATTRAVERO LE CHIVI A CUI SONO ASSOCIATE
# Il metoto SEATDEFAULT fa si che se passiamo una chiave:valore non presente nel dizioanrio lo crea e lo inserisce

mio_dizionario = {"mia chiave uno":"mio valore uno", "eta":24, 3.14:"pi_greco", "primi":[1,2,3,4]}
print(mio_dizionario["mia chiave uno"]) #STAMPIAMO IL VALORE ASSOCIATO A TALE CHIAVA
mio_dizionario["nuova_chiave"] = "nuovo_valore"
mio_dizionario["eta"] = 99
del mio_dizionario["mia chiave uno"]
print(mio_dizionario.get("mia chiave uno")) #STAMPIAM IL VALORE ASSOCIATO A QUELLA CHIAVE PASSATA
print(mio_dizionario.keys()) #STAMPA TUTTE LE KEYS INTERNE AL DIZIONARIO
print(mio_dizionario.values()) #STAMPA TUTTI I VALORI ASSOCIATI ALLE KEY NEL DIZIONARIO
print(mio_dizionario.items()) #STAMPA UNA LISTA FORMATA DA COPPIE (KEY:VALORE)
print(mio_dizionario)

# STRINGHE E FORMATTAZIONE
y = 7
ciao = f"Il valore è {y}"
print(ciao)

#.startswith()
#.endswith()
#.isupper()
#.islower()
