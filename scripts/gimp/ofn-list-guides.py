#!/usr/bin/env python
# -*- coding: utf-8 -*-

# GIMP plugin list guides in current image
# (c) Ofnuts 2020
#
#   History:
#
#   v0.0: 2020-11-04 First (re-?)published version
#
#   This program is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published
#   by the Free Software Foundation; either version 3 of the License, or
#   (at your option) any later version.
#
#   This very file is the complete source code to the program.
#
#   If you make and redistribute changes to this code, please mark it
#   in reasonable ways as different from the original version. 
#   
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   The GPL v3 licence is available at: https://www.gnu.org/licenses/gpl-3.0.en.html

from gimpfu import *

def getGuides(image):
    guides=[[],[]] # ORIENTATION-HORIZONTAL (0), ORIENTATION-VERTICAL (1) 
    gid=0
    while True:
        gid=image.find_next_guide(gid)
        if not gid:
            break;
        guides[image.get_guide_orientation(gid)].append(image.get_guide_position(gid))
    map(lambda x:x.sort(),guides)
    return guides

def list2str(guides):
    return ", ".join([str(g) for g in guides])

def listGuides(image):
    guides=getGuides(image)
    gimp.message("Horizontal guides: %s\nVertical guides: %s" 
                 % (list2str(guides[0]),list2str(guides[1])))

### Registrations
register(
    "ofn-list-guides",
    "List guides in current image",
    "List guides in current image",
    "Ofnuts","Ofnuts","2020",
    "List guides",
    "*",
    [
        (PF_IMAGE, "image", "Input image", None),
    ],
    [],
    listGuides,
    menu="<Image>/Image/Guides"
)

main()
