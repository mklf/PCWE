use clap;
use std::num;
use std::fmt;
use std::error;
use std::io;
use bincode;
#[derive(Debug)]
pub enum W2vError {
    File(io::Error),
    RuntimeError,
    Decode(bincode::rustc_serialize::DecodingError),
}

impl From<io::Error> for W2vError {
    fn from(err: io::Error) -> W2vError {
        W2vError::File(err)
    }
}
impl From<bincode::rustc_serialize::DecodingError> for W2vError {
    fn from(err: bincode::rustc_serialize::DecodingError) -> W2vError {
        W2vError::Decode(err)
    }
}


impl fmt::Display for W2vError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            W2vError::File(ref reason) => write!(f, "open file error:{}", reason),
            W2vError::Decode(ref reason) => write!(f, "decode file error:{}", reason),
            W2vError::RuntimeError => write!(f, "word2vec runtime error"),
        }
    }
}
pub enum ArgumentError {
    ParseArg(clap::Error),
    ParseInt(num::ParseIntError),
    ParseFloat(num::ParseFloatError),
    Other(OtherError),
}

#[derive(Debug)]
pub struct OtherError{
    reason:String,
}

impl fmt::Display for OtherError{
    fn fmt(&self,f:&mut fmt::Formatter) -> fmt::Result {
        write!(f,"ArgumentError:{}",self.reason)
    }
}
impl error::Error for OtherError{
    fn description(&self) -> &str{
        &self.reason
    }
}


impl From<clap::Error> for ArgumentError {
    fn from(err: clap::Error) -> ArgumentError {
        ArgumentError::ParseArg(err)
    }
}
impl From<num::ParseIntError> for ArgumentError {
    fn from(err: num::ParseIntError) -> ArgumentError {
        ArgumentError::ParseInt(err)
    }
}

impl From<num::ParseFloatError> for ArgumentError {
    fn from(err: num::ParseFloatError) -> ArgumentError {
        ArgumentError::ParseFloat(err)
    }
}

impl fmt::Display for ArgumentError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ArgumentError::ParseArg(ref err) => write!(f, "Parse args:{}", err),
            ArgumentError::ParseInt(ref err) => write!(f, "Parse int:{}", err),
            ArgumentError::ParseFloat(ref err) => write!(f, "Parse float:{}", err),
            ArgumentError::Other(ref err) => write!(f,"Error:{}",err),
        }
    }
}
impl fmt::Debug for ArgumentError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ArgumentError::ParseArg(ref err) => write!(f, "Parse args:{:?}", err),
            ArgumentError::ParseInt(ref err) => write!(f, "Parse int:{:?}", err),
            ArgumentError::ParseFloat(ref err) => write!(f, "Parse float:{:?}", err),
            ArgumentError::Other(ref err) => write!(f, "Parse float:{:?}", err),
        }
    }
}
impl error::Error for ArgumentError {
    fn description(&self) -> &str {
        match *self {
            ArgumentError::ParseArg(ref err) => err.description(),
            ArgumentError::ParseFloat(ref err) => err.description(),
            ArgumentError::ParseInt(ref err) => err.description(),
            ArgumentError::Other(ref err) => err.description(),
        }
    }
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            ArgumentError::ParseArg(ref err) => Some(err),
            ArgumentError::ParseFloat(ref err) => Some(err),
            ArgumentError::ParseInt(ref err) => Some(err),
            ArgumentError::Other(ref err) => Some(err),
        }
    }
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Command {
    Train,
    Test,
}


#[derive(Debug,Clone)]
pub struct Argument {
    pub input: String,
    pub output: String,
    pub lr: f32,
    pub dim: usize,
    pub win: usize,
    pub epoch: u32,
    pub neg: usize,
    pub neg_pow:f64,
    pub nthreads: u32,
    pub min_count: u32,
    pub threshold: f32,
    pub lr_update: u32,
    pub min_ngram: usize,
    pub max_ngram: usize,
    pub command: Command,
    pub verbose: bool,
    pub similar_file_path: String,
    pub low_count: u32
}

struct ArgumentBuilder {
    pub input: String,
    pub output: String,
    pub lr: f32,
    pub dim: usize,
    pub win: usize,
    pub epoch: u32,
    pub neg: usize,
    pub nthreads: u32,
    pub min_count: u32,
    pub threshold: f32,
    pub lr_update: u32,
    pub min_ngram: usize,
    pub max_ngram: usize,
    pub command: Command,
    pub verbose: bool,
    pub neg_pow: f64,
    pub similar_file_path: String,
    pub low_count:u32
}
impl ArgumentBuilder {
    pub fn new(input: String, command: Command) -> ArgumentBuilder {
        ArgumentBuilder {
            input: input,
            output: "".to_string(),
            lr: 0.05,
            dim: 100,
            win: 5,
            epoch: 5,
            neg: 5,
            nthreads: 12,
            min_count: 5,
            threshold: 1e-4,
            lr_update: 100,
            command: command,
            verbose: false,
            min_ngram:0,
            max_ngram:0,
            neg_pow:0.5,
            similar_file_path:"".to_owned(),
            low_count:0,
        }
    }
    #[allow(dead_code)]
    pub fn output(&mut self, output: String) -> &mut Self {
        self.output = output;
        self
    }
    #[allow(dead_code)]
    pub fn similar_file_path(&mut self, similar_file_path: String) -> &mut Self {
        self.similar_file_path = similar_file_path;
        self
    }
    #[allow(dead_code)]
    pub fn lr(&mut self, lr: f32) -> &mut Self {
        self.lr = lr;
        self
    }
    #[allow(dead_code)]
    pub fn dim(&mut self, dim: usize) -> &mut Self {
        self.dim = dim;
        self
    }
    #[allow(dead_code)]
    pub fn win(&mut self, win: usize) -> &mut Self {
        self.win = win;
        self
    }
    #[allow(dead_code)]
    pub fn neg_pow(&mut self, neg_pow:f64) -> &mut Self {
        self.neg_pow = neg_pow;
        self
    }
    #[allow(dead_code)]
    pub fn epoch(&mut self, epoch: u32) -> &mut Self {
        self.epoch = epoch;
        self
    }
    #[allow(dead_code)]
    pub fn neg(&mut self, neg: usize) -> &mut Self {
        self.neg = neg;
        self
    }
    #[allow(dead_code)]
    pub fn threads(&mut self, threads: u32) -> &mut Self {
        self.nthreads = threads;
        self
    }
    #[allow(dead_code)]
    pub fn min_count(&mut self, min_count: u32) -> &mut Self {
        self.min_count = min_count;
        self
    }
    #[allow(dead_code)]
    pub fn threshold(&mut self, threshold: f32) -> &mut Self {
        self.threshold = threshold;
        self
    }
    #[allow(dead_code)]
    pub fn lr_update(&mut self, lr_update: u32) -> &mut Self {
        self.lr_update = lr_update;
        self
    }
    #[allow(dead_code)]
    pub fn verbose(&mut self, verbose: bool) -> &mut Self {
        self.verbose = verbose;
        self
    }
    #[allow(dead_code)]
    pub fn min_ngram(&mut self, minn: usize) -> &mut Self {
        self.min_ngram = minn;
        self
    }
    #[allow(dead_code)]
    pub fn max_ngram(&mut self, maxn: usize) -> &mut Self {
        self.max_ngram = maxn;
        self
    }
    #[allow(dead_code)]
    pub fn low_count(&mut self, low_count: u32) -> &mut Self {
        self.low_count = low_count;
        self
    }
    #[allow(dead_code)]
    pub fn finalize(&self) -> Argument {
        Argument {
            input: self.input.to_owned(),
            output: self.output.to_owned(),
            win: self.win,
            epoch: self.epoch,
            lr: self.lr,
            dim: self.dim,
            neg: self.neg,
            nthreads: self.nthreads,
            min_count: self.min_count,
            threshold: self.threshold,
            lr_update: self.lr_update,
            command: self.command,
            verbose: self.verbose,
            min_ngram:self.min_ngram,
            max_ngram:self.max_ngram,
            neg_pow:self.neg_pow,
            similar_file_path:self.similar_file_path.to_owned(),
            low_count: self.low_count
        }
    }
}

pub fn parse_arguments<'a>(args: &'a Vec<String>) -> Result<Argument, ArgumentError> {
    let app = clap_app!(word2vec =>
        (version: "1.0")
        (author: "Frank Lee <golifang1234@gmail.com>")
        (about: "word2vec implemention for rust")
        (@subcommand test =>
        (about: "test word similarity")
        (@arg input:+required "input parameter file path( use train subcommand to train a model)")
        (@arg verbose: --verbose "print internal log")
        )
       (@subcommand train =>
            (about: "train model")
            (version: "0.1")
         //argument
        (@arg input: +required "input corpus file path")
        (@arg output: +required "file name to save params")
        //options
        (@arg win: --win +takes_value "window size(5)")
        (@arg neg: --neg +takes_value "negative sampling size(5)")
        (@arg lr: --lr +takes_value "learning rate(0.05)")
        (@arg lr_update: --lr_update +takes_value "learning rate update rate(100)")
        (@arg dim: --dim +takes_value "size of word vectors(100)")
        (@arg epoch: --epoch +takes_value "number of epochs(5)")
        (@arg min_count: --min_count +takes_value "number of word occurences(5)")
        (@arg min_ngram: --minn +takes_value "min length of ngram (0)")
        (@arg max_ngram: --maxn +takes_value "max length of ngram(0 to disable)")
        (@arg low_count: --lc +takes_value "low word count(0 to disable)")
        (@arg neg_pow: --npow +takes_value "neg_table pow(0.5)")
        (@arg nthreads: --thread +takes_value "number of threads(12)")
        (@arg threshold: --threshold +takes_value "sampling threshold(1e-4)")
        (@arg verbose: --verbose "print internal log")
        (@arg sim_file_path: --sim_file_path + takes_value "file path to nearest neighbor file")
        )
    );
    let matches = app.get_matches();

    if let Some(train_info) = matches.subcommand_matches("train") {
        let input = try!(train_info.value_of("input")
            .ok_or(clap::Error::argument_not_found_auto("input")));
        let output = try!(train_info.value_of("output")
            .ok_or(clap::Error::argument_not_found_auto("output")));
        let win = try!(train_info.value_of("win").unwrap_or("5").parse::<usize>());
        let neg = try!(train_info.value_of("neg").unwrap_or("5").parse::<usize>());
        let lr = try!(train_info.value_of("lr").unwrap_or("0.05").parse::<f32>());
        let lr_update = try!(train_info.value_of("lr_update").unwrap_or("5000").parse::<u32>());
        let vector_size = try!(train_info.value_of("dim").unwrap_or("100").parse::<usize>());
        let epoch = try!(train_info.value_of("epoch").unwrap_or("5").parse::<u32>());
        let min_count = try!(train_info.value_of("min_count").unwrap_or("5").parse::<u32>());
        let min_ngram = try!(train_info.value_of("min_ngram").unwrap_or("0").parse::<usize>());
        let max_ngram = try!(train_info.value_of("max_ngram").unwrap_or("0").parse::<usize>());
        let neg_pow = try!(train_info.value_of("neg_pow").unwrap_or("0.5").parse::<f64>());
        let nthreads = try!(train_info.value_of("nthreads").unwrap_or("12").parse::<u32>());
        let threshold = try!(train_info.value_of("threshold").unwrap_or("1e-4").parse::<f32>());
        let low_count = try!(train_info.value_of("low_count").unwrap_or("0").parse::<u32>());
        let sim_path = train_info.value_of("sim_file_path").unwrap_or("");
        if min_ngram>max_ngram{
            return Err(ArgumentError::Other(OtherError{reason:"minn<=maxn expected".to_owned()}));
        }
        Ok(Argument {
            input: input.to_string(),
            output: output.to_string(),
            lr: lr,
            dim: vector_size,
            win: win,
            epoch: epoch,
            neg: neg,
            nthreads: nthreads,
            min_count: min_count,
            threshold: threshold,
            lr_update: lr_update,
            command: Command::Train,
            verbose: train_info.is_present("verbose"),
            max_ngram:max_ngram,
            min_ngram:min_ngram,
            neg_pow:neg_pow,
            similar_file_path:sim_path.to_string(),
            low_count: low_count
        })
    } else if let Some(ref test_info) = matches.subcommand_matches("test") {
        let input = try!(test_info.value_of("input")
            .ok_or(clap::Error::argument_not_found_auto("input")));
        Ok(ArgumentBuilder::new(input.to_string(), Command::Test)
            .verbose(test_info.is_present("verbose"))
            .finalize())
    } else {
        Err(ArgumentError::ParseArg(clap::Error::argument_not_found_auto("missing arguments")))
    }

}
