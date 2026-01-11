import { Component } from '@angular/core';

@Component({
  selector: 'app-image-processing',
  imports: [],
  templateUrl: './image-processing.component.html',
  styleUrl: './image-processing.component.scss'
})
export class ImageProcessingComponent {
  selectedFile: File | null = null;
  selectedFileName: string | null = null;
  previewUrl: string | ArrayBuffer | null = null;

  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      this.selectedFileName = file.name;
      console.log('File selected:', file.name);

      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        this.previewUrl = reader.result;
      };
      reader.readAsDataURL(file);
    }
  }

  runModels(): void {
    if (!this.selectedFile) {
      alert('Please upload an image first.');
      return;
    }
    console.log('Running models for:', this.selectedFile.name);
  }
}
