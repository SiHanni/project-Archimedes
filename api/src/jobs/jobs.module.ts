import { Module } from '@nestjs/common';
import { PricingModule } from '../pricing/pricing.module';
import { JobAssetsController } from './assets.controller';
import { JobsController } from './jobs.controller';
import { JobsService } from './jobs.service';

@Module({
  imports: [PricingModule],
  controllers: [JobsController, JobAssetsController],
  providers: [JobsService],
})
export class JobsModule {}
