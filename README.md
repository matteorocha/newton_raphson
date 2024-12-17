# newton_raphson

# **Trabalho 2 – Inferência Estatística**  
**Nome:** Matheus Rocha Canto  
**Matrícula:** 22250353  

---

## **Contexto**  
Neste trabalho, explorou-se o uso do método de **Newton-Raphson** para estimar os parâmetros de uma distribuição normal \( (\mu \text{ e } \sigma^2) \), com base em um conjunto de dados fornecido. Os cálculos obtidos foram comparados às estimativas obtidas pela técnica de **Máxima Verossimilhança (EMV)**.

---

## **Definições**  
- **\(\μ\):** Representa a média da distribuição, indicando o valor central. Alterações em \(\mu\) deslocam a distribuição ao longo do eixo horizontal sem modificar sua forma.  
- **\(\sigma^2\):** Representa a variância, que descreve a dispersão dos dados em torno da média. Aumentos em \(\sigma^2\) tornam a curva mais achatada e ampla; reduções tornam a curva mais estreita e alta.

As estimativas foram obtidas ao minimizar:
- Para \(\mu\): A soma dos desvios dos dados em relação à média.  
- Para \(\sigma^2\): A soma dos desvios quadráticos dos dados em relação à média.

---

## **Linguagem Utilizada**  
A linguagem escolhida foi **Python**, com suporte da biblioteca `numpy`, utilizada para realizar cálculos matemáticos e manipular os dados. O conjunto de dados analisado corresponde à coluna **Variável 2**.

---

## **Questões**

### **1) Estimar \(\mu\) com Newton-Raphson e comparar com o EMV da média amostral (\(\bar{x}\))**

**Código em Python:**

```python
import numpy as np

def newton_raphson_mean(x, u0, tol=1e-7, max_iter=100):
    """
    Estima a média populacional (μ) usando o método de Newton-Raphson.

    Parâmetros:
        x (array-like): Dados amostrais.
        u0 (float): Chute inicial para a média.
        tol (float): Tolerância para critério de parada. Padrão: 1e-7.
        max_iter (int): Número máximo de iterações. Padrão: 100.

    Retorna:
        μ (float): Estimativa da média populacional.
        estimativas (list): Histórico das estimativas ao longo das iterações.
    """
    n = len(x)
    u_k = u0
    estimativas = [u_k]

    for i in range(max_iter):
        g_u = np.sum(x - u_k)
        g_prime_u = -n

        if g_prime_u == 0:
            return None, estimativas  # Evita divisão por zero

        u_next = u_k - g_u / g_prime_u
        estimativas.append(u_next)

        if abs(u_next - u_k) < tol:  # Critério de parada
            return u_next, estimativas

        u_k = u_next

    return None, estimativas  # Não convergiu

# Dados fornecidos
dados = np.array([199, 267, 272, 166, 239, 189, 238, 223, 279, 190, 240, 209, 210, 171, 255, 232, 147, 268,
                  231, 199, 255, 199, 228, 240, 184, 192, 211, 201, 203, 243, 181, 382, 186, 198, 165, 219,
                  196, 239, 259, 162, 178, 246, 176, 157, 179, 231, 183, 213, 230, 134, 181, 234, 161, 289,
                  186, 298, 211, 189, 164, 219, 287, 179, 216, 224, 212, 230, 231, 185, 180, 205, 219, 286,
                  261, 221, 194, 248, 216, 195, 217, 186, 218, 173, 221, 206, 215, 176, 240, 234, 190, 204,
                  256, 296, 223, 225, 217, 251, 187, 290, 238, 218])

u0 = 1  # Chute inicial
emv = np.mean(dados)  # Média amostral (EMV)
estimativa, _ = newton_raphson_mean(dados, u0)

print(f"Newton-Raphson: μ = {estimativa:.2f}")
print(f"EMV: μ = {emv:.2f}")
