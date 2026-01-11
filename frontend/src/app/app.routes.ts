import { Routes } from '@angular/router';
import { ImageProcessingComponent } from './shared/pages/image-processing/image-processing.component';

export const routes: Routes = [
    {
        path: '',
        component: ImageProcessingComponent,
        title: 'Image Processing Page',
    }
];
