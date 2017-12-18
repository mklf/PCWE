use {Matrix, Dict};
use std::io::{BufWriter, BufReader};
use std::io::prelude::*;
use std::fs::File;
use bincode;
use bincode::rustc_serialize::{encode_into, decode_from};
use utils::W2vError;
use Model;
pub struct Word2vec {
    syn0: Matrix,
    syn1neg: Matrix,
    dim: usize,
    dict: Dict,
}

impl Word2vec {
    pub fn new(syn0: Matrix, syn1neg: Matrix, dim: usize, dict: Dict) -> Word2vec {
        Word2vec {
            syn0: syn0,
            syn1neg: syn1neg,
            dim: dim,
            dict: dict,
        }
    }
    pub fn norm_self(&mut self) {
        self.syn0.norm_self();
    }
    #[cfg(feature="blas")]
    #[inline(always)]
    pub fn most_similar(&self, word: &str, topn: Option<usize>) -> Vec<(f32, String)> {
        let mut vec = vec![0.;self.dict.nword_size()];
        let c = self.dict.get_idx(word);
        let row = self.syn0.get_row_unmod(c);
        self.syn0.sgemv(row, vec.as_mut_ptr());
        let mut sorted = Vec::new();
        for i in 0..vec.len() {
            sorted.push((vec[i], self.dict.get_word(i)));
        }
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let topn = topn.unwrap_or(10);
        sorted.into_iter().take(topn).collect()
    }
    fn compute_hidden(input:&Vec<usize>,hidden:&mut Vec<f32>,syn0:&mut Matrix){
        Model::zero_vector(hidden);
        for index in input{
            let sub = syn0.get_row(*index);
            Model::add_mul_row(hidden,sub,1.);
        }
        let inv_size = 1./(input.len() as f32);
        let hidden_ptr:*mut f32  = unsafe{ hidden.get_unchecked_mut(0) };
        for i in 0..hidden.len() as isize{
            unsafe{ *hidden_ptr.offset(i) *= inv_size;}
        }

    }
    pub fn transform(mut self)->Self{
        let mut hidden = vec![0f32;self.dim];
        for ent in self.dict.word2ent.values(){
            Word2vec::compute_hidden(&ent.sub_words,&mut hidden,&mut self.syn0);
            unsafe {
                self.syn0.set_row(hidden.get_unchecked(0), ent.index);
            }
        }
        self
    }
    pub fn save_vectors(&self, filename: &str) -> Result<bool, W2vError> {
        let size = self.dict.nword_size();
        let mut file = try!(File::create(filename.to_owned()+".dict"));
        let mut meta = Vec::new();

        try!(write!(&mut meta, "{} {}\n", size, self.dim));
        try!(file.write_all(&meta));
        //let start = self.syn0.get_row_unmod(0);

        for i in 0..size {
            let word = self.dict.get_word(i);
            let freq = self.dict.get_entry(&word);
            try!(file.write(&word.into_bytes()[..]));
            let s = format!(" {}\n",freq.count);
            /*for j in 0..self.dim {
                unsafe {
                    let s = format!(" {}", *start.offset((i * self.dim + j) as isize));
                    try!(file.write(&s.into_bytes()[..]));
                }
            }*/
            try!(file.write(&s.into_bytes()[..]));
        }
        use std::mem;
        use std::slice;
        unsafe{
            let mut file = try!(File::create(filename.to_owned()+".vec"));
            let ptr = self.syn0.get_row_unmod(0);
            let ptr = mem::transmute::<*const f32,*const u8>(ptr);
            let u8data = slice::from_raw_parts(ptr,
                                               size*self.dim*4);
            try!(file.write(u8data));
        };
        Ok(true)
    }
    pub fn save(&self, filename: &str) {
        let file = File::create(filename).unwrap();
        let mut writer = BufWriter::new(file);
        encode_into(&self.dict, &mut writer, bincode::SizeLimit::Infinite).unwrap();
        encode_into(&self.syn0, &mut writer, bincode::SizeLimit::Infinite).unwrap();
        encode_into(&self.syn1neg, &mut writer, bincode::SizeLimit::Infinite).unwrap();
        encode_into(&self.dim, &mut writer, bincode::SizeLimit::Infinite).unwrap();

    }
    pub fn load_from(filename: &str) -> Result<Word2vec, W2vError> {
        let file = try!(File::open(filename));
        let mut reader = BufReader::new(file);
        let dict: Dict = try!(decode_from(&mut reader, bincode::SizeLimit::Infinite));
        let syn0: Matrix = decode_from(&mut reader, bincode::SizeLimit::Infinite).unwrap();
        let syn1neg: Matrix = decode_from(&mut reader, bincode::SizeLimit::Infinite).unwrap();
        let dim: usize = decode_from(&mut reader, bincode::SizeLimit::Infinite).unwrap();
        Ok(Word2vec {
            dict: dict,
            syn0: syn0,
            syn1neg: syn1neg,
            dim: dim,
        })
    }
}