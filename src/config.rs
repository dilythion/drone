use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
  pub animation: Animation,
  pub sound: Sound,
}

impl Config {
    pub fn new() -> Self {
        let contents = std::fs::read_to_string(
            format!("{}/.config/drone/drone.toml", std::env::home_dir().unwrap().display()),
        ).expect("Should have been able to read the file");
        let decoded: Self = toml::from_str(&contents).unwrap();
        println!("{decoded:#?}");
        decoded
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct Animation {
    pub selector: Option<f32>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Sound {
    pub overtones: Option<Vec<f32>>,
    pub drone: Option<f32>,
    pub compression: Option<f32>,
    pub duration: Option<f32>,
}
