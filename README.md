# **Trabalho 2 – Inferência Estatística**  
**Nome:** Matheus Rocha Canto  
**Matrícula:** 22250353  

---

## **Contexto**  
Neste trabalho, explorou-se o uso do método de **Newton-Raphson** para estimar os parâmetros de uma distribuição normal (μ e σ²) , com base em um conjunto de dados fornecido. Os cálculos obtidos foram comparados às estimativas obtidas pela técnica de **Máxima Verossimilhança (EMV)**.

---

## **Definições**  
- **μ:** Representa a média da distribuição, indicando o valor central. Alterações em *x̄* deslocam a distribuição ao longo do eixo horizontal sem modificar sua forma.  
- **σ²:** Representa a variância, que descreve a dispersão dos dados em torno da média. Aumentos em *ô²* tornam a curva mais achatada e ampla; reduções tornam a curva mais estreita e alta.

As estimativas foram obtidas ao minimizar:
- Para *μ*: A soma dos desvios dos dados em relação à média.  
- Para *σ²*: A soma dos desvios quadráticos dos dados em relação à média.

---

## **Linguagem Utilizada**  
A linguagem escolhida foi **Python**, com suporte da biblioteca `numpy`, utilizada para realizar cálculos matemáticos e manipular os dados. O conjunto de dados analisado corresponde à coluna **Variável 2**.

---

## **Questões**

### **1) Usar algoritmo de Newton-Raphson para estimar média (μ) e comparar a estimativa com o EMV da média amostral (x̄).**

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

    # Cálculo da g(û) e a derivada g'(û)
    for i in range(max_iter):
        g_u = np.sum(x - u_k)
        g_prime_u = -n

        if g_prime_u == 0:
            return None, estimativas  # Evita divisão por zero

        # Atualização de Newton-Raphson
        u_next = u_k - g_u / g_prime_u
        estimativas.append(u_next)

        if abs(u_next - u_k) < tol:  # Critério de parada
            return u_next, estimativas

        u_k = u_next

    return None, estimativas  # Não convergiu

# Dados fornecidos da variável 2
dados = np.array([199, 267, 272, 166, 239, 189, 238, 223, 279, 190, 240, 209, 210, 171, 255, 232, 147, 268,
                  231, 199, 255, 199, 228, 240, 184, 192, 211, 201, 203, 243, 181, 382, 186, 198, 165, 219,
                  196, 239, 259, 162, 178, 246, 176, 157, 179, 231, 183, 213, 230, 134, 181, 234, 161, 289,
                  186, 298, 211, 189, 164, 219, 287, 179, 216, 224, 212, 230, 231, 185, 180, 205, 219, 286,
                  261, 221, 194, 248, 216, 195, 217, 186, 218, 173, 221, 206, 215, 176, 240, 234, 190, 204,
                  256, 296, 223, 225, 217, 251, 187, 290, 238, 218])

u0 = 1  # Chute inicial
emv = np.mean(dados)  # Média amostral (EMV)
estimativa, _ = newton_raphson_mean(dados, u0)

# Resutado das estimativas:
print(f"Estimativa μ para Newton-Raphson: {estimativa:.2f}")
print(f"Estimativa μ para o EMV: {emv:.2f}")

````
## *Resultado*

Estimativa de 𝜇 para Newton-Raphson: 216.96

Estimativa de 𝜇 para o EMV: 216.96

---

### **2) Usar algoritmo de Newton-Raphson para estimar σ² (variância real) e comparar com a estimativa EMV ô² (variância estimada).**

**Código em Python:**

```python
import numpy as np

def newton_raphson_variance(x, sigma2_0, tol=1e-7, max_iter=100):
    """
    Estima a variância populacional (σ²) usando o método de Newton-Raphson.

    Parâmetros:
        x (array-like): Dados amostrais.
        sigma2_0 (float): Chute inicial para a variância.
        tol (float): Tolerância para critério de parada. Padrão: 1e-7.
        max_iter (int): Número máximo de iterações. Padrão: 100.

    Retorna:
        σ² (float): Estimativa da variância populacional.
        estimativas (list): Histórico das estimativas ao longo das iterações.
    """
    n, mean_x = len(x), np.mean(x)
    S = np.sum((x - mean_x)**2)
    sigma2_k = sigma2_0
    estimativas = [sigma2_0]

    # Cálculo da h(ô²) e a derivada h(ô²)
    for i in range(max_iter):
        h_sigma2 = -n / (2 * sigma2_k) + S / (2 * sigma2_k**2)
        h_prime_sigma2 = n / (2 * sigma2_k**2) - S / (sigma2_k**3)

        if h_prime_sigma2 == 0:
            return None, estimativas

        # Atualização de Newton-Raphson
        sigma2_next = sigma2_k - h_sigma2 / h_prime_sigma2
        estimativas.append(sigma2_next)

        if abs(sigma2_next - sigma2_k) < tol:
            return sigma2_next, estimativas

        sigma2_k = sigma2_next

    return None, estimativas

# Dados fornecidos da variável 2
dados = np.array([199, 267, 272, 166, 239, 189, 238, 223, 279, 190, 240, 209, 210, 171, 255, 232, 147, 268,
                  231, 199, 255, 199, 228, 240, 184, 192, 211, 201, 203, 243, 181, 382, 186, 198, 165, 219,
                  196, 239, 259, 162, 178, 246, 176, 157, 179, 231, 183, 213, 230, 134, 181, 234, 161, 289,
                  186, 298, 211, 189, 164, 219, 287, 179, 216, 224, 212, 230, 231, 185, 180, 205, 219, 286,
                  261, 221, 194, 248, 216, 195, 217, 186, 218, 173, 221, 206, 215, 176, 240, 234, 190, 204,
                  256, 296, 223, 225, 217, 251, 187, 290, 238, 218])

sigma2_0 = 1  # Chute inicial
emv_sigma2 = np.var(dados, ddof=0)  # Variância não corrigida
estimativa_sigma2, _ = newton_raphson_variance(dados, sigma2_0)

# Resultado da estimativa
print(f"Estimativa de σ² pelo método de Newton-Raphson: {estimativa_sigma2:.2f}")
print(f"Estimativa de σ² pela fórmula EMV: {emv_sigma2:.2f}")


````

## *Resultado*

Estimativa de σ² pelo método de Newton-Raphson: 1494.88

Estimativa de σ² pela fórmula EMV: 1494.88

---

## *Conclusão*

O método de Newton-Raphson utilizado nas duas questões provou eficácia na obtenção dos resultados. Nos dois casos analisados (μ e σ²), os valores convergiram rapidamente para os obtidos pelo EMV, confirmando a precisão de uma estimativa próxima do esperado. A ferramenta numpy foi essencial na aplicação das fórmulas descritas e nos indícios de parâmetros, portanto concluindo o processo do algoritmo.
