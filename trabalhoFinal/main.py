import cv2
import os
from datetime import datetime
import numpy as np
from multiprocessing import Process
import csv


def salvar_resultados_csv(nome_arquivo, k, info_original, info_gerada):
    with open(nome_arquivo.replace(".png", ".csv"), "a", newline="") as csvfile:
        colunas = [
            "K",
            "Tamanho KB Original",
            "Tamanho KB Gerado",
            "Cores unicas originais",
            "Cores unicas geradas",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=colunas)

        # Se o arquivo não existir, escreva os cabeçalhos
        if csvfile.tell() == 0:
            writer.writeheader()

        # Escreve a linha de dados
        writer.writerow(
            {
                "K": k,
                "Tamanho KB Original": info_original[1],
                "Tamanho KB Gerado": info_gerada[1],
                "Cores unicas originais": info_original[2],
                "Cores unicas geradas": info_gerada[2],
            }
        )


def obter_informacoes_imagem(imagem, imagem_path):
    # Obtém resolução, tamanho em KB e quantidade de cores únicas
    resolucao = imagem.shape[:2]
    # Obtém o tamanho real dos dados codificados
    tamanho_kb = os.stat(imagem_path).st_size / 1024.0
    cores_unicas = len(set(tuple(pixel) for linha in imagem for pixel in linha))
    return resolucao, tamanho_kb, cores_unicas


def aplicar_kmeans_e_salvar(imagem, k_values, pasta_saida, nome_original, imagem_path):
    def processar(k):
        # Aplicar o algoritmo K-Means
        pixels = imagem.reshape((-1, 3))
        kmeans = cv2.kmeans(
            pixels.astype(np.float32),
            k,
            None,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
            attempts=10,
            flags=cv2.KMEANS_RANDOM_CENTERS,
        )
        centroides = np.uint8(kmeans[2])

        # Criar imagem a partir dos centroides
        imagem_segmentada = centroides[kmeans[1].flatten()].reshape(imagem.shape)

        # Obter informações sobre as imagens original e gerada
        info_original = obter_informacoes_imagem(imagem, imagem_path)

        # Salvar imagens
        nome_arquivo = f'resultado_k{k}_{"x".join(map(str, info_original[0]))}.jpg'
        cv2.imwrite(
            os.path.join(pasta_saida, nome_arquivo),
            cv2.cvtColor(imagem_segmentada, cv2.COLOR_RGB2BGR),
        )

        # Obter informações sobre as imagens original e gerada
        info_gerada = obter_informacoes_imagem(
            imagem_segmentada, pasta_saida + "/" + nome_arquivo
        )

        # Imprimir informações
        print(f"{nome_original}: K = {k}: {info_original} -> {info_gerada}")
        salvar_resultados_csv(
            pasta_saida + nome_original[1:], k, info_original, info_gerada
        )

    process = []
    for k in k_values:
        proc = Process(target=processar, args=(k,))
        process.append(proc)
        proc.start()

    for proc in process:
        proc.join()


def makedirs(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


if __name__ == "__main__":
    # Lista de caminhos das imagens originais
    imagens = [
        "pictures/1.png",
        "pictures/2.png",
        "pictures/3.png",
        "pictures/4.png",
        "pictures/5.png",
        "pictures/6.png",
    ]

    # Valores de k
    valores_k = [2, 3, 4, 5, 6, 7, 8]

    # Pasta de saída para os resultados
    pasta_saida = makedirs("resultados_kmeans_" + str(datetime.now()))

    procs = []

    for imagem_path in imagens:
        # Ler a imagem original
        imagem_original = cv2.imread(imagem_path)
        imagem_original = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2RGB)

        nome_original = os.path.basename(imagem_path)

        pasta_local = makedirs(
            pasta_saida + "/" + imagem_path.split("/")[1].split(".")[0]
        )

        # Aplicar K-Means e salvar imagens
        proc = Process(
            target=aplicar_kmeans_e_salvar,
            args=(imagem_original, valores_k, pasta_local, nome_original, imagem_path),
        )
        procs.append(proc)
        proc.start()

    # Completar processos
    for proc in procs:
        proc.join()
