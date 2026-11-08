# dados/ (fora do git)

```
dados/
  brutos/publicos/<dataset>/   zips do Roboflow descompactados (data.yaml, train/, valid/, test/)
  brutos/proprias/<trecho>/    fotos ORIGINAIS da caminhada, uma pasta por trecho (com GPS: nunca publicar)
  intermediario/               saída do 01 e do 02 (já remapeado, borrado e sem metadados)
  pre_anotado/                 saída do 03, para subir no Roboflow/CVAT
  anotado/proprias/            export YOLOv8 do Roboflow (ou YOLO 1.1 do CVAT) das suas fotos
  dataset/                     dataset final do 04: train/ val/ test/ + data.yaml
```
