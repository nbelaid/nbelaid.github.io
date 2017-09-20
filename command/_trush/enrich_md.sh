cp ../dev/enron/enron.md ../web/enron/enron.md
echo '' | cat - ../web/enron/enron.md > temp && mv temp ../web/enron/enron.md
echo '---' | cat - ../web/enron/enron.md > temp && mv temp ../web/enron/enron.md
echo 'layout: template1' | cat - ../web/enron/enron.md > temp && mv temp ../web/enron/enron.md
echo '---' | cat - ../web/enron/enron.md > temp && mv temp ../web/enron/enron.md
