import cv2
import os
from datetime import datetime
import numpy as np
from multiprocessing import Process


def obter_informacoes_imagem(imagem):
    # Obtém resolução, tamanho em KB e quantidade de cores únicas
    resolucao = imagem.shape[:2]
    # Obtemos diretamente o tamanho da imagem em bytes usando a função imread
    tamanho_bytes = cv2.imencode(".jpg", imagem)[1].tobytes()
    # Convertendo para kilobytes
    tamanho_kb = len(tamanho_bytes) / 1024.0
    cores_unicas = len(set(tuple(pixel) for linha in imagem for pixel in linha))
    return resolucao, tamanho_kb, cores_unicas


def aplicar_kmeans_e_salvar(imagem, k_values, pasta_saida, nome_original):
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
        info_original = obter_informacoes_imagem(imagem)
        info_gerada = obter_informacoes_imagem(imagem_segmentada)

        # Salvar imagens
        nome_arquivo = f'resultado_k{k}_{"x".join(map(str, info_original[0]))}.jpg'
        cv2.imwrite(
            os.path.join(pasta_saida, nome_arquivo),
            cv2.cvtColor(imagem_segmentada, cv2.COLOR_RGB2BGR),
        )

        # Imprimir informações
        print(f"{nome_original}: K = {k}: {info_original} -> {info_gerada}")

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
            args=(imagem_original, valores_k, pasta_local, nome_original),
        )
        procs.append(proc)
        proc.start()

    # Completar processos
    for proc in procs:
        proc.join()
