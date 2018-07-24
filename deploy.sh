#/bin/bash
echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"
des="public"
msg="rebuilding site `date`"
if [ $# -eq 1  ]
    then msg="$1"
fi

git add -A
git commit -m "$msg"
git push origin master

rm -rf $des/*

# Build the project.
hugo # if using a theme, replace by `hugo -t <yourtheme>`
hugo-algolia --config algolia.yaml -s
cp CNAME $des
cd $des


git add -A
git commit -m "$msg"
git push origin master

