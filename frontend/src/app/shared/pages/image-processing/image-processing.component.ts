import { Component, inject } from '@angular/core';
import { ApiService, PredictionResult, PretrainedPredictionResult } from '../../services/api.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-image-processing',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './image-processing.component.html',
  styleUrl: './image-processing.component.scss'
})
export class ImageProcessingComponent {
  private apiService = inject(ApiService);

  selectedFile: File | null = null;
  selectedFileName: string | null = null;
  previewUrl: string | ArrayBuffer | null = null;

  customResult: PredictionResult['result'] | null = null;
  pretrainedResult: PretrainedPredictionResult['results'][0] | null = null;
  isLoading = false;

  onFileSelected(event: any): void {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      this.selectedFileName = file.name;

      // Reset results
      this.customResult = null;
      this.pretrainedResult = null;

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

    this.isLoading = true;

    // Call Pretrained Model
    this.apiService.predictPretrained(this.selectedFile).subscribe({
      next: (res) => {
        if (res.results && res.results.length > 0) {
          this.pretrainedResult = res.results[0];
        }
      },
      error: (err) => console.error('Pretrained Error', err)
    });

    // Call Custom Model
    this.apiService.predictCustom(this.selectedFile).subscribe({
      next: (res) => {
        this.customResult = res.result;
        this.isLoading = false;
      },
      error: (err) => {
        console.error('Custom Error', err);
        this.isLoading = false;
      }
    });
  }
}
