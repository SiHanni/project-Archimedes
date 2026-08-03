import { BadRequestException, Controller, Get, Query } from '@nestjs/common';
import { PricingService } from './pricing.service';

@Controller('pricing')
export class PricingController {
  constructor(private readonly pricing: PricingService) {}

  /** 프런트가 시세만 따로 보고 싶을 때 (견적 자체는 job 결과에 붙어 나온다) */
  @Get('spot')
  async spot(@Query('metal') metal?: string, @Query('purity') purity?: string) {
    if (!metal || !purity) {
      throw new BadRequestException('metal and purity are required');
    }
    const value = await this.pricing.getSpot(metal.toLowerCase(), purity.toLowerCase());
    if (!value) {
      return { available: false, metal, purity, buyRate: this.pricing.buyRate };
    }
    return { available: true, ...value, buyRate: this.pricing.buyRate };
  }
}
