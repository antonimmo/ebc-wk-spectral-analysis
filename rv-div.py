## Dados:
## rAz, rAc (áreas de las caras de las diferentes mallas)
## dxG, dyG, dxC, dyC (diferenciales de longitud en las diferentes mallas)
## U,V (componentes del vector) --- matriz 3 ejes (t=[0,-3],filas[1,-2],columnas[2,-1]) --> filas ("dirección" y), columnas ("dirección" x)

## Axis 

DIV = (np.gradient(dyG*U,axis=-1,edge_order=2) + np.gradient(dxG*V,axis=-2,edge_order=2))/rAc
RV = (np.gradient(dyC*V,axis=-1,edge_order=2) - np.gradient(dxC*U,axis=-2,edge_order=2))/rAz

##
## Si el tiempo no es el primer índice de las matrices U,V, sino el último --> (filas[0,-3],columnas[1,-2],t[2,-1]), hay 2 opciones

## 1. Indicar en np.gradient los ejes correctos
DIV = (np.gradient(dyG*U,axis=1,edge_order=2) + np.gradient(dxG*V,axis=0,edge_order=2))/rAc
RV = (np.gradient(dyC*V,axis=1,edge_order=2) - np.gradient(dxC*U,axis=0,edge_order=2))/rAz

## 2. "Mover" temporalmente los ejes para que el tiempo quede al final, y usar la misma función "original": (filas[0,-3],columnas[1,-2],t[2,-1]) --> (t=[0,-3],filas[1,-2],columnas[2,-1])
U_ = np.moveaxis(U, -1, 0) ## "Vista" de la matriz original, cualquier cambio en U_ se verá reflejado en U
V_ = np.moveaxis(V, -1, 0)
DIV_ = (np.gradient(dyG*U_,axis=1,edge_order=2) + np.gradient(dxG*V_,axis=0,edge_order=2))/rAc
RV_ = (np.gradient(dyC*V_,axis=1,edge_order=2) - np.gradient(dxC*U_,axis=0,edge_order=2))/rAz
## Regresamos a la misma forma que tenían U,V de entrada
DIV = np.moveaxis(DIV_, 0, -1)
RV = np.moveaxis(RV_, 0, -1)
