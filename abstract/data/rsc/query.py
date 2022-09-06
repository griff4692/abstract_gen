
query = 'https://pubs.rsc.org/en/results/journals?Category=Journal&IncludeReference=false&SelectJournal=true&ArticleType=Paper&DateRange=false&SelectDate=false&Type=Months&PriceCode=False&OpenAccess=true'


import scrapy


class PaperSpider(scrapy.Spider):
    name = 'quotes'
    start_urls = [query]

    def parse(self, response):
        print('\n\n\n\n\n\n\n\n\n\n\n')
        print([x for x in response.css('a').xpath('@href').getall() if 'article' in x])
        print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
        for url in response.css("a.btn"):
            print('\n\n\n\n\n')
            print(url)
            print('\n\n\n\n\n')
            yield {
                'href': url.extract()
            }

        next_page = response.css('li.paging__item a').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)


# url = 'https://pubs.rsc.org/en/content/articlepdf/2022/fd/d2fd00022a'
