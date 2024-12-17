# **Trabalho 2 – Inferência Estatística**  

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


### **2) Usar algoritmo de Newton-Raphson para estimar σ² (variância real) e comparar com a estimativa EMV ô² (variância estimada).**

