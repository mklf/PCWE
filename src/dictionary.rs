use std::io::{BufReader, stdout};
use std::io::prelude::*;
use std::fs::File;
use std::collections::HashMap;
use NEGATIVE_TABLE_SIZE;
use rand::{thread_rng, Rng};
use rand::distributions::{IndependentSample, Range};
use std::sync::Arc;
use super::W2vError;
use std::ops::Index;
use std::usize;
#[derive(RustcEncodable, RustcDecodable, PartialEq,Debug)]
pub struct Dict {
    pub word2ent: HashMap<String, Entry>,
    pub idx2word: Vec<String>,
    pub ntokens: usize,
    size: usize,
    discard_table: Vec<f32>,
    pub subwords:HashMap<String,usize>,//n-gram
}

#[derive(RustcEncodable, RustcDecodable, PartialEq,Debug)]
pub struct Entry {
    pub index: usize,
    pub count: u32,
    pub sub_words:Vec<usize>,
    pub similar_words:Vec<usize>,
}


impl Dict {
    fn new() -> Dict {
        Dict {
            word2ent: HashMap::new(),
            idx2word: Vec::new(),
            ntokens: 0,
            size: 0,
            discard_table: Vec::new(),
            subwords:HashMap::new()
        }
    }
    pub fn init_negative_table(&self,neg_pow:f64) -> Arc<Vec<usize>> {
        let mut negative_table = Vec::new();
        let counts = self.counts();
        let mut z = 0f64;
        for c in &counts {
            z += (*c as f64).powf(neg_pow);
        }
        for (idx, i) in counts.into_iter().enumerate() {
            let c = (i as f64).powf(neg_pow);
            for _ in 0..(c * NEGATIVE_TABLE_SIZE as f64 / z) as usize {
                negative_table.push(idx as usize);
            }
        }
        let mut rng = thread_rng();
        rng.shuffle(&mut negative_table);
        Arc::new(negative_table)

    }

    fn add_to_dict(words: &mut HashMap<String, Entry>, word: &str, size: &mut usize) {
        words.entry(word.to_owned())
            .or_insert_with(|| {
                let ent = Entry {
                    index: *size,
                    count: 0,
                    sub_words:Vec::new(),
                    similar_words:Vec::new(),
                };
                *size += 1;
                ent
            })
            .count += 1;
    }
    //fasttext里面在每个词上加了一个BOS,EOS，考察一下是否要加上？
    pub fn add_ngrams(&mut self,min:usize,max:usize,verbose:bool){
        let min = if min==0 {min+1} else{ min };
        let mut index:isize = -1;
        let bos = String::from("<");
        let eos = ">";
        let (mut count,word_size) = (0,self.nword_size());
        for (key,val)  in self.word2ent.iter_mut(){
            let chars:Vec<char> = (bos.clone()+key+eos).chars().collect();
            //先把这个词对应的向量下标放进去
            val.sub_words.push(val.index);
            if max==0{continue}
            let len:usize = chars.len();
            for i in 0..len{
                let candidate_end = i+max+1;
                let end = if len+1<candidate_end{ len+1 } else {candidate_end};
                for j in (i+min)..end{
                    count +=1;
                    let sub_str = chars[i..j].iter().cloned().collect();
                    let inserted_index = self.subwords.entry(sub_str).or_insert_with(
                        || {
                            index+=1;
                            index as usize +word_size
                        }
                    );
                    val.sub_words.push(*inserted_index);
                }
            }
        }
        if verbose{
            println!("#uniq ngrams:{} #total ngrams:{}",index+1,count);
        }
    }


    pub fn add_subwords(&mut self,min:usize,max:usize,verbose:bool){
        self.add_ngrams(min,max,verbose);
        self.subwords.shrink_to_fit();
    }

    pub fn add_similar_words(&mut self,similar_file_path:String,
                             verbose:bool)->Result<(), W2vError>{
        if similar_file_path ==""{
            return Ok(());
        }
        let input_file = try!(File::open(similar_file_path));
        let mut reader = BufReader::new(input_file);
        let mut buf_str = String::new();
        let mut valid_similar = 0;
        let mut center_count = 0;
        while reader.read_line(&mut buf_str).unwrap()>0{
            let words:Vec<String> = buf_str.split_whitespace().into_iter().
                map(|s| s.to_owned()).collect();
            let mut indexes = Vec::new();
            for w in &words{
                match self.word2ent.get(w){
                    Some(ent) => {
                        indexes.push(ent.index);
                    }
                    None => {
                        indexes.push(usize::max_value());
                    }
                }
            }

            let entry= self.word2ent.get_mut(&words[0]);

             match entry{
                Some(ent) => {
                    for index in &indexes[1..]{
                        if *index!= usize::max_value() {
                            ent.similar_words.push(*index);
                            valid_similar +=1;
                        }
                    }
                    center_count+=1;
                }
                None => {}
            }
            buf_str.clear();
        }
        if verbose{
            println!("#words with neighbor:{},#neighbor words {}",
                     center_count,valid_similar);
        }
        Ok(())

    }

    pub fn nword_size(&self) -> usize {
        self.size
    }
    pub fn  ngram_size(&self) ->usize{
        self.subwords.len()
    }

    pub fn n_vectors(&self) -> usize{
        // 词的数目+ (ngram 数目+ 其他特征数目)
        self.size + self.subwords.len()
    }
    #[inline(always)]
    pub fn get_idx(&self, word: &str) -> usize {
        self.word2ent[word].index
    }
    #[inline(always)]
    pub fn get_word(&self, idx: usize) -> String {
        self.idx2word[idx].clone()
    }
    #[inline(always)]
    pub fn get_entry_by_idx(&self, idx: usize) -> &Entry{
        self.word2ent.get(&self.idx2word[idx]).unwrap()
    }
    #[inline(always)]
    pub fn get_entry(&self,word:&str)->&Entry{
        self.word2ent.index(word)
    }
    pub fn counts(&self) -> Vec<u32> {
        let mut counts_ = vec![0;self.idx2word.len()];
        for (i, v) in self.idx2word.iter().enumerate() {
            counts_[i] = self.word2ent[v].count;
        }
        counts_
    }
    pub fn read_line<'a>(&'a self, line: &mut String, lines: &mut Vec<&'a Entry>) -> usize {
        let mut i = 0;
        let mut rng = thread_rng();
        let between = Range::new(0., 1.);
        for word in line.split_whitespace() {
            i += 1;
            match self.word2ent.get(word) {
                Some(e) => {
                    if self.discard_table[e.index] > between.ind_sample(&mut rng) {
                        lines.push(e);
                    }
                }
                None => {}
            }
        }
        i
    }
    pub fn new_from_file(filename: &str,
                         min_count: u32,
                         threshold: f32,
                         verbose: bool)
                         -> Result<Dict, W2vError> {
        let mut dict = Dict::new();
        let input_file = try!(File::open(filename));
        let mut reader = BufReader::with_capacity(10000, input_file);
        let mut buf_str = String::with_capacity(5000);
        let mut words: HashMap<String, Entry> = HashMap::with_capacity(2<<20);
        let (mut ntokens, mut size) = (0, 0);
        while reader.read_line(&mut buf_str).unwrap() > 0 {
            for word in buf_str.split_whitespace() {
                Dict::add_to_dict(&mut words, word, &mut size);
                ntokens += 1;
                if ntokens % 1000000 == 0 {
                    print!("\rRead {}M words", ntokens / 1000000);
                    stdout().flush().ok().expect("Could not flush stdout");
                }
            }
            buf_str.clear();
        }
        size = 0;
        let word2ent: HashMap<String, Entry> = words.into_iter()
            .filter(|&(_, ref v)| v.count >= min_count)
            .map(|(k, mut v)| {
                v.index = size;
                size += 1;
                (k, v)
            })
            .collect();
        dict.word2ent = word2ent;
        dict.word2ent.shrink_to_fit();
        dict.idx2word = vec!["".to_string();dict.word2ent.len()];
        for (k, v) in &dict.word2ent {
            dict.idx2word[v.index] = k.to_string();
        }
        dict.idx2word.shrink_to_fit();
        dict.size = size;
        dict.ntokens = ntokens;
        if verbose {
            println!("\rRead {} M words", (ntokens / 1000000));
            println!("\r{} unique words in total", size);

        }
        dict.init_discard(threshold);
        Ok(dict)
    }
    fn init_discard(&mut self, threshold: f32) {
        let size = self.nword_size();
        self.discard_table.reserve_exact(size);
        for i in 0..size {
            let f = self.word2ent[&self.idx2word[i]].count as f32 / self.ntokens as f32;
            self.discard_table.push((threshold / f).sqrt() + threshold / f);
        }
    }
}