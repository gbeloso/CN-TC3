import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import sympy as sp
    import math

    return (np,)


@app.cell
def _(np):
    # Preenchimento da tabela de diferenças divididas
    def get_matriz_diferencas(x, y, matriz=None):
        n = len(x)
        old_n = 0 if matriz is None else matriz.shape[0]

        if matriz is None:
            matriz = np.zeros((n, n), dtype=np.float64)
            matriz[:, 0] = y
        elif n > old_n:
            # Redimensiona e preserva os valores anteriores
            matriz = np.pad(
                matriz,
                pad_width=((0, n - old_n), (0, n - old_n)),
                mode='constant',
                constant_values=0,
            )
            # Atualiza a primeira coluna com todos os pontos y
            matriz[:, 0] = y
        else:
            raise ValueError(
                "Numero de pontos passados menor ou igual ao tamanho da matriz passada"
            )

        # O laço de colunas (i) vai de 1 até n-1.
        # O laço de linhas (j) começa em max(i, old_n) para reaproveitar cálculos passados.
        for i in range(1, n):
            for j in range(max(i, old_n), n):
                f = (matriz[j, i - 1] - matriz[j - 1, i - 1]) / (
                    x[j] - x[j - i]
                )
                matriz[j, i] = f

        return matriz

    return (get_matriz_diferencas,)


@app.function
def get_polinomio(matriz, x):
    n = len(x)

    def p(X):
        resultado = matriz[0, 0]
        for i in range(1, n):
            m = matriz[i, i]
            for j in range(0, i):
                m *= X - x[j]
            resultado += m
        return resultado

    return p


@app.cell
def _(np):
    def get_coeficientes_canonicos(matriz, x):
        n = len(x)

        p_total = np.array([matriz[0, 0]], dtype=np.float64)
        termo_acumulado = np.array([1.0], dtype=np.float64)

        for i in range(1, n):
            # A convolução com [-x[i-1], 1.0] equivale a multiplicar por (X - x[i-1])
            termo_acumulado = np.convolve(termo_acumulado, [-x[i - 1], 1.0])

            # Multiplica pelo coeficiente da diagonal a_i
            termo_ponderado = matriz[i, i] * termo_acumulado

            # Ajusta o tamanho de p_total com zeros para poder somar vetores de tamanhos diferentes
            if len(p_total) < len(termo_ponderado):
                p_total = np.pad(
                    p_total, (0, len(termo_ponderado) - len(p_total))
                )

            p_total += termo_ponderado

        return p_total

    return (get_coeficientes_canonicos,)


@app.function
def formatar_polinomio(coefs, var='x', decrescente=True):
    """Formata um vetor [c0, c1, c2, ...] em uma string polinomial legível.

    - coefs: lista ou array [c0, c1, ..., cn]
    - var: nome da variável
    - decrescente: se True, exibe da maior potência para a menor
    """
    if not any(coefs):
        return f"P({var}) = 0"

    # Mapeamento para expoentes visuais em Unicode (x², x³, etc.)
    sobrescritos = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

    termos = []
    graus_coefs = list(enumerate(coefs))
    if decrescente:
        graus_coefs.reverse()

    for grau, coef in graus_coefs:
        # Ignora coeficientes nulos
        if abs(coef) < 1e-12:
            continue

        sinal = "-" if coef < 0 else "+"
        val_abs = abs(coef)

        # Formata o número (remove decimais desnecessários se for inteiro)
        val_str = f"{val_abs:.4g}"

        # Monta a parte da variável e expoente
        if grau == 0:
            termo_var = ""
        elif grau == 1:
            termo_var = var
            if val_str == "1":
                val_str = ""  # Exibe 'x' em vez de '1x'
        else:
            exp_unicode = str(grau).translate(sobrescritos)
            termo_var = f"{var}{exp_unicode}"
            if val_str == "1":
                val_str = ""  # Exibe 'x²' em vez de '1x²'

        termos.append((sinal, f"{val_str}{termo_var}"))

    if not termos:
        return f"p({var}) = 0"

    # Junta os termos com espaçamento correto nos sinais
    primeiro_sinal, primeiro_termo = termos[0]
    resultado = f"-{primeiro_termo}" if primeiro_sinal == "-" else primeiro_termo

    for sinal, termo in termos[1:]:
        resultado += f" {sinal} {termo}"

    return f"p({var}) = {resultado}"


@app.cell
def _(get_coeficientes_canonicos, get_matriz_diferencas):
    def problema_1():
        print('='*70)
        print("Problema 1:\n")
        x = [1.2, 2.1, 3, 3.6]
        y = [0.7, 8.1, 27.7, 45.1]

        matriz = get_matriz_diferencas(x, y)
        print("Matriz de Diferenças Divididas:")
        print(matriz)
        print()

        p_teste = get_polinomio(matriz, x)
        print("Teste nos pontos originais:")
        for xi in x:
            print(f"p({xi}) = {p_teste(xi)}")

        coefs = get_coeficientes_canonicos(matriz, x)

        print(formatar_polinomio(coefs))
        print("\nFim do problema 1")
        print('='*70, '\n')


    return (problema_1,)


@app.cell
def _(get_coeficientes_canonicos, get_matriz_diferencas, np):
    def problema_2():
        print('='*70)
        print("Problema 2:\n")
        x = [1,2,3,4,5,6]
        y = [14.5,19.5,30.5,53.5,94.5,159.5]

        matriz = get_matriz_diferencas(x, y)
        print("Matriz de Diferenças Divididas:")
        print(matriz)
        print()

        p_teste = get_polinomio(matriz, x)
        print("Teste nos pontos originais:")
        for xi in x:
            print(f"p({xi}) = {p_teste(xi)}")

        coefs = get_coeficientes_canonicos(matriz, x)

        print(formatar_polinomio(coefs))

        print(f'\np(4.5) interpolado: {p_teste(4.5)}, erro absoluto: {np.abs(p_teste(4.5) - 71.375)}')

        print("\nFim do problema 2")

        print('='*70, '\n')

    return (problema_2,)


@app.cell
def _(get_coeficientes_canonicos, get_matriz_diferencas, np):
    def problema_3():
        print('='*70)
        print("Problema 3:\n")
        x = [0, 10, 20, 30, 40, 60, 80, 100]
        y = [0.0061, 0.0123, 0.0234, 0.0424, 0.0738, 0.1992, 0.4736, 1.0133]

        matriz = get_matriz_diferencas(x, y)
        print("Matriz de Diferenças Divididas:")
        print(matriz)
        print()

        p_teste = get_polinomio(matriz, x)
        print("Teste nos pontos originais:")
        for xi in x:
            print(f"p({xi}) = {p_teste(xi)}")

        coefs = get_coeficientes_canonicos(matriz, x)

        print(formatar_polinomio(coefs))

        print(f'\np(5) interpolado: {p_teste(5)}, erro absoluto: {np.abs(p_teste(5) - 0.008721)}')
        print(f'p(45) interpolado: {p_teste(45)}, erro absoluto: {np.abs(p_teste(45) - 0.095848)}')
        print(f'p(95) interpolado: {p_teste(95)}, erro absoluto: {np.abs(p_teste(95) - 0.84528)}, aqui fica perceptivel que extrapolar com graus altos eh perigoso')

        print("\nFim do problema 3")

        print('='*70, '\n')


    return (problema_3,)


@app.cell
def _(get_coeficientes_canonicos, get_matriz_diferencas, np):
    def problema_4():
        print('='*70)
        print("Problema 4:\n")
        x = [605, 685, 725, 765, 825, 855, 875]
        y = [0.622, 0.655, 0.668, 0.679, 0.730, 0.907, 1.336]

        matriz = get_matriz_diferencas(x, y)
        print("Matriz de Diferenças Divididas:")
        print(matriz)
        print()

        p_teste = get_polinomio(matriz, x)
        print("Teste nos pontos originais:")
        for xi in x:
            print(f"p({xi}) = {p_teste(xi)}")

        coefs = get_coeficientes_canonicos(matriz, x)

        print(formatar_polinomio(coefs))

        print(f'\np(645) interpolado: {p_teste(645)}, erro absoluto: {np.abs(p_teste(645) - 0.639)}')
        print(f'p(795) interpolado: {p_teste(795)}, erro absoluto: {np.abs(p_teste(795) - 0.694)}')
        print(f'p(845) interpolado: {p_teste(845)}, erro absoluto: {np.abs(p_teste(845) - 0.812)}')

        print("\nFim do problema 4")

        print('='*70, '\n')


    return (problema_4,)


@app.cell
def _(problema_1, problema_2, problema_3, problema_4):
    problema_1()
    problema_2()
    problema_3()
    problema_4()
    return


if __name__ == "__main__":
    app.run()
