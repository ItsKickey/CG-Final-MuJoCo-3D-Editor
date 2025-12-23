# A Mujoco obj loader

## requirements
OpenGL >3.3
python >3.11

## how to use 
``` bash
pip install -r requirements.txt
python src/main_final.py
```
## Import obj / change floor texture
Due to limitation of Mujoco
please make sure the texture file is .png (not .jpg .jpeg .hdr .tga etc.)

## Loading scene
I don't recommand loading a random sceme due to the data directory in this project
Instead the loading scene is design for "exported scene " with this project
since all obj data must be under Import_mjcf/ (see export scene chapter)

## Export scene
when exporting scene there will be a {filename}.zip generate under outputfile/
as 
    outputfile/
    └── {filename}.zip
        ├── scene/ (Current Scene XML)
        └── Import_mjcf/ (All OBJ Data & Assets)

## known bugs
Since I fully reconstruct the main_final.py and gui.py to replaced Tkinter with ImGui (see Change log 1.1)
there are some bugs  I still can't deal with 
but the main bug that might terminate the project has been remove

Known bug lists: v1.2.1
When import a new obj , the project won't automatically select the imported object imediately and thuse cause physical collision immediately (if there where anything at spawn point)

## Acknowledgments

This project integrates code from obj2mjcf by Kevin Zakka.

Source: https://github.com/kevinzakka/obj2mjcf

License: MIT License