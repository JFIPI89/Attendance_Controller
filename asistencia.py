import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime

#creaer base de datos
ruta = 'Empleados'
mis_imagenes = []
nombres_empleados =[]
lista_empleados = os.listdir(ruta)

#DISTANCIA PERMITIDA
distancia_permitida = 0.6

# bucle para extraer nombres de lista sin extencion del archivo 'jpeg'

for nombre in lista_empleados:
    imagen_actual = cv2.imread(f'{ruta}/{nombre}')
    mis_imagenes.append(imagen_actual)
    nombres_empleados.append(os.path.splitext(nombre)[0])

print(nombres_empleados)

#codigicar imagenes

def codificar(imagenes):

    #crear una lista nueva
    lista_codificada = []
    #pasar todas las imagenes a RGB
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        #codificar
        codificado = fr.face_encodings(imagen)[0]

        #agregar a la lista
        lista_codificada.append(codificado)

    #devolver lista codificada
    return lista_codificada

#registrar los ingresos
def registrar_ingresos(persona):
    f = open('registro.csv', 'r+')
    lista_datos = f.readlines()
    nombres_registro = []
    for linea in lista_datos:
        ingreso = linea.split(',')
        nombres_registro.append(ingreso[0])
    if persona not in nombres_registro:
        ahora = datetime.now()
        string_ahora = ahora.strftime('%H:%M:%S')
        f.writelines(f'\n {persona}, {string_ahora}')




lista_empleados_codificada = codificar(mis_imagenes)

#TOMAR IMAGENES DESDE CAMARA

captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#LEER IMAGEN TOMADA

exito, imagen = captura.read()
if not exito:
    print('ERROR')
else:
    #reconocer rostro de captura
    cara_captura = fr.face_locations(imagen)

    #codificar cara capturada
    cara_captura_codificada = fr.face_encodings(imagen, cara_captura)

    #BUSCAR COINCIDENCIAS

    for caracdodificada, carauubicacion in zip(cara_captura_codificada, cara_captura):
        coincidecnias = fr.compare_faces(lista_empleados_codificada, caracdodificada)
        distancias = fr.face_distance(lista_empleados_codificada, caracdodificada)

        print(distancias)

        indice_coincidencia = numpy.argmin(distancias)

        #MOSTRAR COINCIDENCIAS

        if distancias[indice_coincidencia] > distancia_permitida:
            print('NO COINCIDE CON NINGUN COLABORADOR')
        else:
            #BUSCAR NOMBRE DE COLABORADOR ENCONTRADO
            nombre = nombres_empleados[indice_coincidencia]

            y1, x2, y2, x1 = carauubicacion
            cv2.rectangle(imagen, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(imagen, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(imagen, nombre, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),2)
 #llamar funcion pára registro de ingresos

            registrar_ingresos(nombre)
            #mostrar imagen WEB

            cv2.imshow('IMAGEN WEB', imagen)

            #mantener ventana abierta
            cv2.waitKey()


