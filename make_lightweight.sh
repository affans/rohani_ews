#!/bin/sh



lw=$1"-lightweight"
echo $lw
mkdir $lw
 

ls $1  | grep -v 'data' | xargs -I % cp $1"/"% -t  $lw
