# Tarea 2 - OpenCL y CUDA

Este código corresponde a la Tarea 2 - OpenCL y CUDA de Nicolás Escobar y Vicente Muñoz, alumnos del curso CC7515 Computación en GPU del semestre otoño de 2025. Esta tarea trata sobre implementar El Juego de la Vida de Conway, tanto en CPU como en GPU haciendo uso de CUDA y OpenCL en C++, para luego comparar su desempeño en los distintos entornos y además elegir 2 variaciones que impactan su rendimiento y medir su desempeño. Las variaciones escogidas fueron:
1. Usar tamaños de bloque tanto múltiplos de 32 como no múltiplos de 32. En particular se usan tamaños de bloque 30, 32 y 35.
2. Usar arreglos de dos dimensiones en vez de un mapeo a arreglo de una dimensión.

## Herramientas necesarias

### g++
Es necesario tener instalado el compilador GNU g++ versión 14.2.0, el cual se puede instalar mediante [MSYS2](https://www.msys2.org/) en Windows o mediante el gestor de paquetes que corresponda en Linux:


Debian / Ubuntu / Pop!_OS / Linux Mint
```bash
sudo apt update
sudo apt install build-essential
```
Arch Linux / Manjaro
```bash
sudo pacman -S --needed base-devel
```
Fedora / RHEL / CentOS (8+) / Rocky Linux / AlmaLinux
```bash
sudo dnf groupinstall "Development Tools"
``` 
CentOS 7
```bash
sudo yum groupinstall "Development Tools"
```
openSUSE
```bash
sudo zypper install -t pattern devel_basis
```
Gentoo
```bash
sudo emerge --ask sys-devel/gcc sys-devel/make sys-devel/binutils
```
Void Linux
```bash
sudo xbps-install -S base-devel
```
Alpine Linux
```bash
sudo apk add build-base
```

En macOS se puede obtener mediante [Homebrew](https://brew.sh/)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install make gcc g++
```
### CMake
Para compilar esta librería se necesita CMake 3.31, este se puede descargar desde [la página de CMake](https://cmake.org/download/) o mediante el gestor de paquetes (no se recomienda, pues no se puede asegurar la versión correcta de CMake).\
Si se está en Windows podría ser necesario agregar la carpeta `bin` de la instalación de CMake a la variable de entorno `$PATH` de forma manual.

### CUDA

Es necesario instalar el toolkit de CUDA para correr los ejemplos de CUDA, esto varía según el sistema operativo anfitrión.

Linux: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

Windows: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html


## Configuración

Esta libería se debe compilar en Windows con el generador `MinGW Makefiles` y en Linux/macOs con el generador `Unix Makefiles` usando el compilador GNU g++ version 14.2.0\
Para ello se deben configurar las siguientes variables de entorno:
+ CC: Ruta al compilador de lenguaje C, por defecto con una instalación estándar de MSYS2 es `C:\msys64\ucrt64\bin\gcc.exe`, en linux `/usr/bin/gcc`.
+ CXX: Ruta al compilador de lenguaje C++, por defecto con una instalación estándar de MSYS2 es `C:\msys64\ucrt64\bin\g++.exe`, en linux `/usr/bin/g++`.
+ CMAKE_GENERATOR: `"MinGW Makefiles"` o `"Unix Makefiles"` según corresponda.

En el caso de CUDA en Windows, es necesario utilizar el generador y compilador de Visual Studio, por lo que estas variables de entorno debiesen tener los siguientes valores:
+ CC: Ruta del ejecutable `cl.exe`, si está instalado Visual Studio, `cl.exe` debiese ser parte de la variable de entorno `PATH`, por lo que basta con solo colcar `cl.exe`
+ CXX: Lo mismo que arriba.
+ CMAKE_GENERATOR: `"Visual Studio 17 2022"`.

Hay otras variables de entorno necesarias que se definen dentro de la terminal de Visual Studio, principalmente:
+ CudaToolkitDir: Path a la instalación de los binarios de CUDA, por defecto en Windows debiese ser `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y` donde `X.Y` es la versión de CUDA a utilizar, en las pruebas se utilizó CUDA v12.9
+ CUDA_PATH: Mismo que arriba
+ CUDA_PATH_X_Y: Mismo que arriba con X_Y la version de CUDA, en el caso de las pruebas 12_9.

Ninguno de los 2 autores de esta tarea utiliza un computador que no sea Windows, pero en Linux, CUDA necesita utilizar el compilador `nvcc` y el generador debisese simplemente ser `Unix Makefiles`.

Para dejar estas variables configuradas durante la sesión actual de terminal se debe ingresar lo siguiente en su terminal de preferencia.

### cmd
```
> set CC=C:\msys64\ucrt64\bin\gcc.exe
> set CXX=C:\msys64\ucrt64\bin\g++.exe
> set CMAKE_GENERATOR="MinGW Makefiles"
```
### powershell
```pwsh
> $env:CC="C:\msys64\ucrt64\bin\gcc.exe"
> $env:CXX="C:\msys64\ucrt64\bin\g++.exe"
> $env:CMAKE_GENERATOR="MinGW Makefiles"
```
### bash
```bash
$ export CC=/usr/bin/gcc
$ export CXX=/usr/bin/g++
$ export CMAKE_GENERATOR="Unix Makefiles"
```

De no configurarse dichas variables de entorno, se pueden también configurar para una invocación única de CMake mediante los siguientes argumentos:
```bash
$ cmake -DCC=/usr/bin/gcc -DCXX=/usr/bin/g++ -DCMAKE_GENERATOR="Unix Makefiles" <comando>
```
(Reemplazar los valores de cada variable según el sistema operativo que corresponda).

Además dependiendo de que implementación se desea compilar, es necesario configurar al menos 1 de las siguientes 3 variables de entorno:
```
-DUSE_CPU=true
-DUSE_CUDA=true
-DUSE_OPENCL=true
```
La primera permitirá compilar la implementación en CPU del juego de la vida, la segunda la implementación en GPU con CUDA y la tercera, con OpenCL.

Adicionalmente existe otra variable que se le puede entregar a CMake junto con `-DUSE_CPU` que es `-DUSE_PARALLEL` para compilar una versión paralela en CPU del juego de la vida.

Notar que OpenCL está incluído como submodulos de este repositorio, por lo que, para compilar con OpenCL es necesario clonar el repositorio con
```
git clone --recursive https://github.com/Tchy258/t2-gpu.git
```

Ciertos parámetros del programa están definidos como macros para tener mayor eficiencia en memoria, tales como el tamaño de la grilla, el tamaño de bloque y si se usan arreglos de 2 dimensiones o de una dimension.

Estos valores se pueden editar de 2 formas, ya sea definiendolos como variables al momento de compilar con CMake, o editando el archivo [include/constants.h](include/constants.h).

# Compilación
Habiendo hecho la configuración basta con ejecutar los siguientes comandos en una terminal estando en la carpeta raíz del projecto:
```bash
$ cmake -DUSE_TARGET=true -S . -B build/target
$ cmake --build build/target
```
Donde `TARGET` debe ser reemplazado por `CPU`, `CUDA` u `OPENCL`, (equivalentemente en minúscula para `target`) según corresponda.

Esto generará los ejecutables correspondientes según la variante deseada en la carpeta `build/target`.

Alternativamente si se usa VSCode, están disponibles varios presets en [CMakePresets.json](CMakePresets.json) para cada objetivo.

# Ejecución

Para ejecutar el juego se dispone de ejemplos pequeños, los cuales se asume serán compilados con tamaños de grilla reducidos (como 10 x 10), estos se encuentran en [cpu_serial_example.cpp](src/cpu_serial_example.cpp), [cpu_parallel_example.cpp](src/cpu_parallel_example.cpp), los cuales reciben como argumento la cantidad de iteraciones que se desea realizar, a modo de ejemplo, la implementación en CPU de manera serial se puede ver ejecutando el archivo `build/cpu/CPU/CPUSerialExample` (con la extensión que corresponda) y la cantidad de iteraciones de la manera siguiente:
```bash
$ ./CPUSerialExample k
```
Donde `k` es el número de iteraciones.

Los otros ejecutables de ejemplo se prueban de manera similar. Notar que por defecto estos vienen comentados en el [CMakeLists.txt de src](src/CMakeLists.txt)

Las pruebas fueron realizadas con el código disponible en el archivo [benchmark.cpp](src/benchmark.cpp).

Estos ejecutables para las pruebas, como el que se compilará para `build/cpu/CPUSerialBenchmark` también aceptan un tercer parámetro con un nombre de archivo al cual escribirán las estadísticas resultantes de correr el programa en formato CSV. Un ejemplo de esto sería
```bash
$ ./CPUSerialBenchmark 16 serial.csv
```
Aquí los resultados se guardarán en la carpeta actual (desde donde se ejecute el binario) en el archivo `serial.csv` para 16 iteraciones. Para guardarlo en otra carpeta se debe ingresar una ruta absoluta o relativa.

Para facilitar la tarea de compilación y ejecución de todas las pruebas, se dispone del archivo [run_presets.py](run_presets.py) que compila y ejecuta uno por uno todos los presets contenidos en [CMakePresets.json](CMakePresets.json), este incluye presets base para CPU, CUDA y OpenCL junto con una gran cantidad de variaciones según tamaño de grilla y de bloque. Para evitar factores externos, tales como archivos cacheados, este script siempre recompila antes de ejecutar algún preset. Para ejecutarlo, solo se necesita Python3 sin ningún paquete adicional y el siguiente comando:
```bash
$ python3 ./run_presets.py
```

Si se desean generar presets con distintos tamaños de bloque o grilla, se dispone del archivo [generate_presets.py](generate_presets.py) que genera un CMakePresets.json

## Análisis

El procesado de los datos generados se hace mediante el jupyter notebook disponible en [analysis.ipynb](analysis.ipynb), este notebook necesita correr en un entorno de python que tenga los paquetes contenidos en [requirements.txt](requirements.txt), teniendo un entorno virtual (o conda) activado, se pueden instalar estas dependencias mediante el comando
```bash
$ pip install -r requirements.txt
```