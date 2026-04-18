import numpy as np


# ************ 
# DISTRIBUCION GAUSSIANA 
# ************ 

# def generar_datos_3d_3Nubes(numero_puntos=100):
#     """Vamos a generar dos nubes de puntos en 3D, cada una con 100 particulas, con una media diferente y una desviación estándar de 1."""

#     x_nube1, y_nube1, z_nube1 = np.random.multivariate_normal(
#     mean=[3, 3, 3], 
#     cov=[[1, -0.3, 0.3], [-0.3, 1, -0.3], [0.3, -0.3, 1]],
#     size=numero_puntos).T

#     x_nube2, y_nube2, z_nube2 = np.random.multivariate_normal(
#         mean=[9,9,9], 
#         cov=[[1, -0.3, 0.3], [-0.3, 1, -0.3], [0.3, -0.3, 1]],
#         size=numero_puntos).T
    
#     x_nube3, y_nube3, z_nube3 = np.random.multivariate_normal(
#         mean=[3,9,9], 
#         cov=[[1, -0.3, 0.3], [-0.3, 1, -0.3], [0.3, -0.3, 1]],
#         size=numero_puntos).T

#     return x_nube1, y_nube1, z_nube1, x_nube2, y_nube2, z_nube2, x_nube3, y_nube3, z_nube3, np.array([1]*numero_puntos + [0]*(2*numero_puntos)), np.array([0]*numero_puntos + [1]*numero_puntos + [0]*numero_puntos)



# ************ 
# DISTRIBUCION UNIFORME 
# ************
# def generar_datos_3d_3Nubes(numero_puntos=100):
#     x1 = np.random.uniform(2, 4, numero_puntos)
#     y1 = np.random.uniform(2, 4, numero_puntos)
#     z1 = np.random.uniform(2, 4, numero_puntos)

#     x2 = np.random.uniform(5, 7, numero_puntos)
#     y2 = np.random.uniform(5, 7, numero_puntos)
#     z2 = np.random.uniform(5, 7, numero_puntos)

#     x3 = np.random.uniform(2, 4, numero_puntos)
#     y3 = np.random.uniform(5, 7, numero_puntos)
#     z3 = np.random.uniform(5, 7, numero_puntos)

#     return x1,y1,z1,x2,y2,z2,x3,y3,z3, \
#            np.array([1]*numero_puntos + [0]*(2*numero_puntos)), \
#            np.array([0]*numero_puntos + [1]*numero_puntos + [0]*numero_puntos)


# ************ 
# DISTRIBUCION EXPONENCIAL 
# ************
# def generar_datos_3d_3Nubes(numero_puntos=100):
#     x1 = np.random.exponential(1, numero_puntos) + 2
#     y1 = np.random.exponential(1, numero_puntos) + 2
#     z1 = np.random.exponential(1, numero_puntos) + 2

#     x2 = np.random.exponential(1, numero_puntos) + 10
#     y2 = np.random.exponential(1, numero_puntos) + 10
#     z2 = np.random.exponential(1, numero_puntos) + 10

#     x3 = np.random.exponential(1, numero_puntos) + 2
#     y3 = np.random.exponential(1, numero_puntos) + 8
#     z3 = np.random.exponential(1, numero_puntos) + 8
#     return x1,y1,z1,x2,y2,z2,x3,y3,z3, np.array(
#         [[1, 0, 0]] * numero_puntos +
#         [[0, 1, 0]] * numero_puntos +
#         [[0, 0, 1]] * numero_puntos
#     )


# ************ 
# DISTRIBUCION NORMAL SIN COVARIANZA  - NORMAL INDEPENDIENTE
# ************
def generar_datos_3d_3Nubes(numero_puntos=100):
    x1 = np.random.normal(3, 1, numero_puntos)
    y1 = np.random.normal(3, 1, numero_puntos)
    z1 = np.random.normal(3, 1, numero_puntos)

    x2 = np.random.normal(9, 1, numero_puntos)
    y2 = np.random.normal(9, 1, numero_puntos)
    z2 = np.random.normal(9, 1, numero_puntos)

    x3 = np.random.normal(3, 1, numero_puntos)
    y3 = np.random.normal(9, 1, numero_puntos)
    z3 = np.random.normal(9, 1, numero_puntos)

    return x1,y1,z1,x2,y2,z2,x3,y3,z3, np.array(
        [[1, 0, 0]] * numero_puntos +
        [[0, 1, 0]] * numero_puntos +
        [[0, 0, 1]] * numero_puntos
    )




# def generar_datos_3d_3Nubes_binomial(numero_puntos=100):
#     n, p = 10, 0.5  # parámetros de la binomial

#     x1 = np.random.binomial(n, p, numero_puntos)
#     y1 = np.random.binomial(n, p, numero_puntos)
#     z1 = np.random.binomial(n, p, numero_puntos)

#     x2 = np.random.binomial(n, p, numero_puntos) + 10
#     y2 = np.random.binomial(n, p, numero_puntos) + 10
#     z2 = np.random.binomial(n, p, numero_puntos) + 10

#     x3 = np.random.binomial(n, p, numero_puntos)
#     y3 = np.random.binomial(n, p, numero_puntos) + 10
#     z3 = np.random.binomial(n, p, numero_puntos) + 10

#     return x1, y1, z1, x2, y2, z2, x3, y3, z3, \
#            np.array([1]*numero_puntos + [0]*(2*numero_puntos)), \
#            np.array([0]*numero_puntos + [1]*numero_puntos + [0]*numero_puntos)