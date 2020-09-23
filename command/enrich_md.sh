# command use example: sh enrich_md.sh "my_use_case"
arg1="$1"

cp ../dev/$arg1/$arg1.md ../web/$arg1/index.md

echo '' | cat - ../web/$arg1/index.md > temp && mv temp ../web/$arg1/index.md
echo '---' | cat - ../web/$arg1/index.md > temp && mv temp ../web/$arg1/index.md
echo 'layout: template1' | cat - ../web/$arg1/index.md > temp && mv temp ../web/$arg1/index.md
echo '---' | cat - ../web/$arg1/index.md > temp && mv temp ../web/$arg1/index.md
