set -x

parsh="parallel --jobs 4 sh -c {}"

tool_dir=~/

ALIGN_WIDTH=1040
STITCH_WIDTH=1040
ICON_WIDTH=512

DIR=images_align
[ ! -d $DIR ] && (mkdir -p $DIR; ls images_orig | awk '{print "convert -resize '$ALIGN_WIDTH'x images_orig/"$1" '$DIR'/"$1}' | $parsh)

DIR=images_stitch
[ ! -d $DIR ] && (mkdir -p $DIR; ls images_align | awk '{print "convert -resize '$STITCH_WIDTH'x images_align/"$1" '$DIR'/"$1}' | $parsh)

DIR=images_icon
[ ! -d $DIR ] && (mkdir -p $DIR; ls images_stitch | awk '{print "convert -resize '$ICON_WIDTH'x images_stitch/"$1" '$DIR'/"$1}' | $parsh)
