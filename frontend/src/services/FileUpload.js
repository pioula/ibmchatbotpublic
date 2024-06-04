import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Button, CustomInput, FormGroup, Label } from 'reactstrap';
const HOST = "34.171.68.155";
// const HOST = "localhost";

const FileUpload = () => {
  const [file, setFile] = useState(null);

  const [label, setLabel] = useState('Wybierz plik PDF');

  useEffect(() => {
    console.log("HETOOO")
    if (!file) {
      setLabel('Wybierz plik PDF');
    } else {
      setLabel(file.name);
    }
  }, [label]);

  const handleFileChange = (event) => {
    console.log("HELLOOOOOOOO")
    setFile(event.target.files[0]);
    setLabel(event.target.files[0].name);
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Dodaj najpierw plik');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post("http://" + HOST + ":8000/upload", formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('File uploaded successfully:', response.data);
      setFile(null)
      setLabel('Wybierz plik PDF')
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  return (
    <div>
      {label === 'Wybierz plik PDF' ? <FormGroup>
              <CustomInput
              type="file"
              id="exampleCustomFileBrowser"
              name="customFile"
              label={'Wybierz plik PDF'}
              onChange={handleFileChange} />
            </FormGroup> :
            <><FormGroup>
            <CustomInput
            type="file"
            id="exampleCustomFileBrowser"
            name="customFile"
            label={file.name}
            onChange={handleFileChange} />
            </FormGroup>
            <div></div></>
            
          }
      <Button color="info" onClick={handleUpload}>Zapisz</Button>
    </div>
  );
};

export default FileUpload;