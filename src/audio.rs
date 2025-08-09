use cpal::{
    traits::{DeviceTrait, HostTrait},
    SizedSample,
};
use cpal::{FromSample, Sample};

fn sine(phase: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    let value = (phase * two_pi).sin();
    assert!(value <= 1.0);
    value
}

fn softened_volume(time: f32) -> f32 {
    match time {
        x if x < 0.2 => x / 0.2,
        x if x > 1.8 => (1.0 - (x - 1.8) / 0.2).max(0.0),
        _ => 1.0,
    }
}

pub struct Oscillation {
    pub sample_rate: f32,
    pub phase: f32,
    pub time: f32,
    pub frequency: f32,
}

impl Oscillation {
    fn new(sample_rate: f32, frequency: f32) -> Self {
        let mut oscillation = Self { 
            sample_rate,
            frequency: 0.0,
            time: 0.0,
            phase: 0.0,
        };
        oscillation.set_frequency(frequency);
        oscillation
    }

    fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency;
        assert!(self.frequency > 50.0 && self.frequency < 20_000.0);
    }

    fn tick(&mut self) -> f32 {
        self.phase = (self.phase + self.frequency / self.sample_rate) % 1.0;
        self.time += 1.0 / self.sample_rate;
        sine(self.phase) 
        + sine(self.phase * 2.0) * 0.2
        + sine(self.phase * 3.0) * 0.5
        + sine(self.phase * 4.0) * 0.1
    }
}

pub fn stream_setup_for(rx: std::sync::mpsc::Receiver<crate::Request>) -> Result<cpal::Stream, anyhow::Error> {
    let (_host, device, config) = host_device_setup()?;

    type Format = cpal::SampleFormat;
    match config.sample_format() {
        Format::I8  => make_stream::<i8>    (&device, &config.into(), rx),
        Format::I16 => make_stream::<i16>   (&device, &config.into(), rx),
        Format::I32 => make_stream::<i32>   (&device, &config.into(), rx),
        Format::I64 => make_stream::<i64>   (&device, &config.into(), rx),
        Format::U8  => make_stream::<u8>    (&device, &config.into(), rx),
        Format::U16 => make_stream::<u16>   (&device, &config.into(), rx),
        Format::U32 => make_stream::<u32>   (&device, &config.into(), rx),
        Format::U64 => make_stream::<u64>   (&device, &config.into(), rx),
        Format::F32 => make_stream::<f32>   (&device, &config.into(), rx),
        Format::F64 => make_stream::<f64>   (&device, &config.into(), rx),
        sample_format => Err(anyhow::Error::msg(format!("unsupported sample format {sample_format}"))),
    }
}

pub fn host_device_setup(
) -> Result<(cpal::Host, cpal::Device, cpal::SupportedStreamConfig), anyhow::Error> {
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::Error::msg("Default output device is not available"))?;
    println!("Output device : {}", device.name()?);

    let config = device.default_output_config()?;
    println!("Default output config : {config:?}");

    Ok((host, device, config))
}

pub fn make_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    rx: std::sync::mpsc::Receiver<crate::Request>,
) -> Result<cpal::Stream, anyhow::Error>
where
    T: SizedSample + FromSample<f32>,
{
    let num_channels = config.channels as usize;
    let mut drone = Oscillation::new(config.sample_rate.0 as f32, 440.0);    
    let err_fn = |err| eprintln!("error building output sound stream: {err}");

    let mut notes: Vec<Oscillation> = vec![];
    let sample_rate = config.sample_rate.0 as f32;

    let stream = device.build_output_stream(
        config,
        move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
            for i in (0..notes.len()).rev() {
                if notes[i].time > 2.0 {
                    notes.remove(i);
                }
            }
            while let Ok(request) = rx.try_recv() {
                match request {
                    crate::Request::AddNote(frequency) => notes.push(Oscillation::new(sample_rate, frequency)),
                    crate::Request::SetDrone(frequency) => drone.set_frequency(frequency),
                }
            }
            process_frame(output, &mut notes, num_channels, &mut drone)
        },
        err_fn,
        None,
    )?;

    Ok(stream)
}

fn process_frame<SampleType>(
    output: &mut [SampleType],
    notes: &mut Vec<Oscillation>,
    num_channels: usize,
    drone: &mut Oscillation,
) where
    SampleType: Sample + FromSample<f32>,
{
    for frame in output.chunks_mut(num_channels) {
        let mut pressure = drone.tick(); 
        for note in &mut *notes {
            pressure += softened_volume(note.time) * note.tick();
        }
        pressure *= 0.2;
        pressure = pressure.min(1.0);
        let value: SampleType = SampleType::from_sample(pressure);
        for sample in frame.iter_mut() { *sample = value; }
    }
}
