docker run -it --rm --env JEKYLL_ENV=production --volume="D:\ÔÓÏî\²©¿Í\blog:/srv/jekyll" jekyll/jekyll jekyll build

scp -Cr ./_site root@txy:/home