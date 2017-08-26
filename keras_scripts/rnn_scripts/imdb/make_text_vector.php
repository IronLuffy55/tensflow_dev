#!/usr/bin/php
<?php
if($file = $argv[1]) {
  if(!file_exists($file)) die("$file does not exist");
  $content = file_get_contents($file);
} else { 
  $content = file_get_contents("php://stdin"); // die("Missing file name");
}
if(!($max_index = $argv[2]) || ($max_index < 0)) $max_index = 0;
if($max_index)echo "Ignore values > $max_index\n";

$words = array_map(function($word) { 
  return rtrim(trim($word),'.,'); 
}, preg_split("/\s/", $content));


$imdbwords = e55_json_decode(file_get_contents('imdb_word_index2.json')); 

$arr = [];

array_walk($words, function($word) use ($imdbwords, &$arr) {
  $lc = strtolower($word); 
  $ival = $imdbwords[$lc];
  if($ival) {
    $arr[] = $ival; 
  }
});
/*
//echo "File content: $content\n";
foreach($words as $word){
  $cmd = "grep -w \"\\\"$word\\\"\" imdb_word_index2.json |  awk -F: '{ print $2 }' | awk -F, '{ print $1 }'";
  $val = `$cmd`;
  $i = intval($val);
  if($max_index && ($i > $max_index)) continue;
  $arr[] = $i;
}
*/
//echo "Array:\n";
echo e55_json_encode($arr);

