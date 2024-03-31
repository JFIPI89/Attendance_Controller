import cv2
import face_recognition as fr

#cargar imagenes

foto_control = fr.load_image_file('image6.jpeg')
foto_prueba = fr.load_image_file('image1.jpeg')

#tolerancia de reconozimiento

tol = 0.55

#pasar imagenes de BRG a RGB

foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)

#LOCALIZAR CARA CONTROL
lugar_cara_A = fr.face_locations(foto_control)[0]
cara_codificada_A = fr.face_encodings(foto_control)[0]

#LOCALIZAR CARA DE PRUEBA
lugar_cara_B = fr.face_locations(foto_prueba)[0]
cara_codificada_B = fr.face_encodings(foto_prueba)[0]

#Mostrar rectangulos

cv2.rectangle(foto_control,
              (lugar_cara_A[3], lugar_cara_A[0]),
              (lugar_cara_A[1], lugar_cara_A[2]),
              (0,255,0)
              ,2)

cv2.rectangle(foto_prueba,
              (lugar_cara_B[3], lugar_cara_B[0]),
              (lugar_cara_B[1], lugar_cara_B[2]),
              (0,255,0)
              ,2)

#comparacion de imagenes

resultado = fr.compare_faces([cara_codificada_A], cara_codificada_B, tol)


# MEDIDA DE LA DISTANCIA

distancia = fr.face_distance([cara_codificada_A], cara_codificada_B)

# MOSTRAR RESULTADO EN PANTALLA

cv2.putText(foto_prueba,
            f'{resultado} {distancia.round(2)}',
            (50,50),
            cv2.FONT_HERSHEY_COMPLEX,
            1,
            (0, 255, 0),
            2)


#MOSTRAR IMAGENES

cv2.imshow('foto de control', foto_control)
cv2.imshow('foto de prueba', foto_prueba)




#mantener programa corriendo
cv2.waitKey(0)


