import { Routes } from '@angular/router';
import { ImageProcessingComponent } from './shared/pages/image-processing/image-processing.component';
import {ComparisonComponent} from "./shared/pages/comparisonpage/comparisonpage";

export const routes: Routes = [
    {
        path: '',
        component: ImageProcessingComponent,
        title: 'Image Processing Page',
    },
    {
        path: 'comparison',
        component: ComparisonComponent
    }
];
