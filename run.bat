docker run -it --rm --env JEKYLL_ENV=production --volume="D:\����\����\blog:/srv/jekyll" jekyll/jekyll jekyll build

scp -Cr ./_site root@txy:/home