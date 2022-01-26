# Rotina de rastreamento de pessoas e faces em vídeo

## Informações

Utiliza as técnicas inteligẽncia computacional (DNN) para triar pessoas e faces em arquivos de vídeo.

__autor:__ Adelino Pinheiro Silva
__email:__ adelinocpp@yahoo.com

Adaptado do trabalho de __Zihao Zhang__ no [github](https://github.com/zzh8829/yolov3-tf2), email: zzh8829@gmail.com e utiliza do [face recognition](https://pypi.org/project/face-recognition/)  em python

As rotinas foram testadas e depuradas com Anaconda 4.10.3 e python 3.8 utilizando a IDE Spyder rodando em _linux Mint 19.2_. __Não realizei nenhum teste de funcionamento ou implementação__ para rodar em _Windows_.

### Instruções

Crie um diretório de trabalho e nele descompacte os arquivos com as rotinas de detecção.

Ainda no diretório de trabalho crie um subdiretório de nome "VIDEOS" e um segundo denome "FACES" da seguinte forma:

```
../
	/VIDEOS
	/FACES
	P00_Detecta_Pessoas_Faces_v4.py
	...
	P02_Agrega_Frames_Pessoas_lista_v0.py
```
No diretŕorio "VIDEOS" despeje os arquivos do tipo *.avi e *.mp4 em que deseja realizar as buscas. Particularmente desaconcelho arquicos compactados pelo codes H.264 e variantes (AVC- Advanced Video Codec) pela dificuldade da biblioteca cv2 (computer vision) de realizar saltos de frames em vídeos longos (ainda explico isso)

No diretório "Faces" armazene a(s) face(s) de referência para busca junto com algumas faces quaisquer, apenas para ficarem como distratores. As imagens podem ser do tipo .png, *.jpg ou *.jpeg.

#### Sobre os nomes dos arquivos das faces

As rotinas foram planejadas para buscar várias faces e para suportar interrupções de processamento. Desta forma alguns protocolos precisam ser seguidos na nomenclatura dos arquivos que apresentam as faces a serem buscadas.

O nome do arquivo de imagem dete possuir o padrão "FFFF_stringtag_vNN", onde FFFF é o numero que identifica a face, NN o numero que identifica a variante da face e "stringtag" é uma etiqueta de identificação que aparecerá nos resultados. Exemplos:
 
1. Desejo buscar a face de um único indivíduo chamado "Homero Gaio Silva", tenho apenas uma foto. Posso nomear o arquivo de imagem como "0001_HGSilva_v01.jpg", é a versão 01 (uncia) da face 0001 (única) com tag "HGSIlva".

2. Deseja-se buscar a face de três indivíduos, "Homero Gaio Silva" com 4 fotos, "Magaret Boiadeiro Silva" com 2 imagens e "Bartolomeu Gaio Silva' com 3 imagens. Pode-se arbitrar a ordem dos indivíduos e os arquivos da forma:
   - "0001_BGSilva_v01.png"
   - "0001_BGSilva_v02.jpg"
   - "0001_BGSilva_v03.png"
   - "0002_HGSilva_v01.png"
   - "0002_HGSilva_v02.jpg"
   - "0002_HGSilva_v03.jpeg"
   - "0002_HGSilva_v04.jpeg"
   - "0003_MBSilva_v01.jpg"
   - "0003_MBSilva_v02.jpg"

Os demais arquivos podem, perferencialmente, ter um nome únoc sem o caracter "_", pois ele é utilizado para separar as informações nos arquivos usados como referência.

### Modelo

O modelo para reconhecimento facial está contido no arquivo "resnet50_coco_best_v2.0.1.h5" que pode ser baixado no [gitlab](https://gitlab.com/pratikjain/sigmoid_test/blob/master/resnet50_coco_best_v2.0.1.h5) ou neste link do [Amazon Drive](https://www.amazon.com/clouddrive/share/6zdD1BnrTZAW2tIfBLzgrY8n0db9czzszF3bWRumrk0).

Descompacte o arquivo "resnet50_coco_best_v2.0.1.h5" no diretório de trabalho.

### Executando a rotina

Execute na sequencia, aguardando a etapa anterior terminar, oas rotinas em ordem

```
$ python3 P00_Detecta_Pessoas_Faces_v4.py
```

em seguida

```
$ python3 P01_Agrega_Frames_Pessoas_v0.py
```

e por fim

```
$ python3 P02_Agrega_Frames_Pessoas_lista_v0.py
```

Em construção... Em breva mais detalhes

### Interpretando resultados

Em construção...