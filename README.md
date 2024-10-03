# Video Labeling Using SAM 2 for interactive segmentation

**Installation**: Follow the [installation instructions](https://github.com/facebookresearch/segment-anything-2#installation) in the SAM2 repository.

### Program Execution

The program parameters are:

```
video2SAM2.py [-h] --input_folder INPUT_FOLDER [--label_colors LABEL_COLORS]
                     [--load_folder LOAD_FOLDER] [--output_folder OUTPUT_FOLDER]
                     [--backup_folder BACKUP_FOLDER] [--resize_factor RESIZE_FACTOR] 
                     [--init_frame INIT_FRAME] [--end_frame END_FRAME] 
                     [--sam_model_folder SAM_MODEL_FOLDER]
                     [--model_size {tiny,small,base_plus,large}]
```

* **--input_folder**: Directory of JPEG frames with filenames like `<frame_index>.jpg`.

    If it is necessary, you can extract their JPEG frames using ffmpeg (https://ffmpeg.org/) as follows:
    ```
    ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%06d.jpg'
    ```
    where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks ffmpeg to start the JPEG file from `000000.jpg`.


* **--load_folder**: Directory from which masks can be loaded when starting the program. Default is `annotations/`.
* **--label_colors**: File containing the class and color information (in RGB) for segmentation. Default is `label_colors.txt`. It follows the [KITTI labeling format](https://docs.cvat.ai/docs/manual/advanced/formats/format-kitti/).
* **--output_folder**: Directory where masks can be saved at the end of the program. Default is `annotations/`.
* **--backup_folder**: Directory where backup copies of the work can be stored. Default is `backups/`.
* **--resize_factor**: Value in $\(0, 1\]$. The dimensions of the image are multiplied by this factor, and a smaller image is used.
* **--init_frame**: Initial frame of the video to process, including this value. Default: $0$.
* **--end_frame**: Final frame of the video, including this value. Value $-1$ for the entire video. Default: $-1$.
* **--sam_model_folder**: Directory where the SAM model is loaded/saved. Default is `models/`.
* **--model_size**: Model sizes of SAM 2 model. Possible values are tiny, small, base_plus, or large.

The controls are displayed in the console.

**Note**: When loading a set of masks, it is necessary that the colors used in them match those in the file with the color information.

**Note 2**: Bounding boxes of the objects can be exported. These will be loaded in the file <load_folder>/bboxes. The format is as follows:
```
class_name instance_number bbox_x bbox_y bbox_w bbox_h
cup 1 220 330 50 65
yellow_duck 2 100 280 40 60
...
```

# Etiquetado de Video Usando SAM 2 para Segmentación Interactiva

**Instalación**: Sigue las [instrucciones de instalación](https://github.com/facebookresearch/segment-anything-2#installation) en el repositorio de SAM2.

### Ejecución del Programa

Los parámetros del programa son:

```
video2SAM2.py [-h] --input_folder INPUT_FOLDER [--label_colors LABEL_COLORS]
                     [--load_folder LOAD_FOLDER] [--output_folder OUTPUT_FOLDER]
                     [--backup_folder BACKUP_FOLDER] [--resize_factor RESIZE_FACTOR] 
                     [--init_frame INIT_FRAME] [--end_frame END_FRAME] 
                     [--sam_model_folder SAM_MODEL_FOLDER]
                     [--model_size {tiny,small,base_plus,large}]
```

* **--input_folder**: Directorio de las imágenes JPEG con nombres de archivo como `<frame_index>.jpg`. 
    
    Si es necesario, puedes extraer las imágenes JPEG utilizando ffmpeg (https://ffmpeg.org/) de la siguiente manera:
    ```
    ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%06d.jpg'
    ```
    donde `-q:v` genera la imagen JPEG de alta calidad y `-start_number 0` indica a ffmpeg que comience el archivo JPEG desde `000000.jpg`.

    
* **--load_folder**: Directorio desde el cual se pueden cargar máscaras al iniciar el programa. El valor predeterminado es `annotations/`.
* **--label_colors**: Archivo que contiene la información de clase y color (en RGB) para la segmentación. El valor predeterminado es `label_colors.txt`. Sigue el formato de etiquetado [KITTI](https://docs.cvat.ai/docs/manual/advanced/formats/format-kitti/).
* **--output_folder**: Directorio donde se pueden guardar las máscaras al final del programa. El valor predeterminado es `annotations/`.
* **--backup_folder**: Directorio donde se pueden almacenar copias de seguridad del trabajo. El valor predeterminado es `backups/`.
* **--resize_factor**: Valor en $\(0, 1\]$. Las dimensiones de la imagen se multiplican por este factor y se trabaja con una imagen más pequeña.
* **--init_frame**: Frame inicial del video, incluyendo este valor. Por defecto: $0$.
* **--end_frame**: Frame final del video, incluyendo este valor. Valor $-1$ para el video completo. Por defecto: $-1$.
* **--sam_model_folder**: Directorio donde se carga/guarda el modelo SAM. El valor predeterminado es `models/`.
* **--model_size**: Tamaño del modelo de SAM 2. Los posibles valores son tiny, small, base_plus o large.

Los controles se muestran en la consola.

**Nota**: Al cargar un conjunto de máscaras, es necesario que los colores utilizados en ellas coincidan con los del archivo con la información de colores.

**Nota 2**: Se pueden exportar las bounding boxes de los objetos. Estas se cargarán en el archivo `<load_folder>/bboxes`. El formato es el siguiente:
```
class_name instance_number bbox_x bbox_y bbox_w bbox_h
cup 1 220 330 50 65
yellow_duck 2 100 280 40 60
...
```
