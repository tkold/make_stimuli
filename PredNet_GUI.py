import Tkinter as tk

root = tk.Tk()
root.title(u'window')

# window size
root.geometry('400x300')

def addList(text):
    ListBox1.insert(tk.END, text)
    print(bln.get())

def deleteSelectedList():
    ListBox1.delete(tk.ACTIVE)


bln = tk.BooleanVar()
bln.set(True)

chk = tk.Checkbutton(variable=bln, text='save')
chk.place(x=50, y=70)

# label
Static1 = tk.Label(text=u'test', foreground='#ff0000', background='#ffaacc')
Static1.pack()

Entry1 = tk.Entry()
Entry1.insert(tk.END, u'Input path')
Entry1.pack()

Button1 = tk.Button(text=u'Button', width=10, command=lambda: addList(Entry1.get()))
Button1.pack()

Button2 = tk.Button(text=u'Delete', width=10, command=lambda: deleteSelectedList())
Button2.pack()

ListBox1 = tk.Listbox()
ListBox1.pack()

root.mainloop()