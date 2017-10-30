#sh enrich_md.sh "my_proj"
arg1="$1"

cp ../dev/$arg1/$arg1.md ../web/$arg1/$arg1.md
#cat ../web/$arg1/_title.md ../dev/$arg1/$arg1.md > ../web/$arg1/$arg1.md

echo '' | cat - ../web/$arg1/$arg1.md > temp && mv temp ../web/$arg1/$arg1.md
echo '---' | cat - ../web/$arg1/$arg1.md > temp && mv temp ../web/$arg1/$arg1.md
echo 'layout: template1' | cat - ../web/$arg1/$arg1.md > temp && mv temp ../web/$arg1/$arg1.md
echo '---' | cat - ../web/$arg1/$arg1.md > temp && mv temp ../web/$arg1/$arg1.md
