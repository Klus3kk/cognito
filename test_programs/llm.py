class DataProcessor:
    def __init__(self, data_source):
        self.source = data_source
        self.processed = False
        self.results = None
    
    def process(self):
        '''
        This is a placeholder for actual data processing
        in a real scenario, this would do something with self.source
        '''
        if not self.processed:
            self.results = [x * 2 for x in self.source if isinstance(x, (int, float))]
            self.processed = True
        return self.results
    
    def get_stats(self):
        if not self.processed:
            self.process()
        
        if not self.results:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "avg": None
            }
        
        return {
            "count": len(self.results),
            "min": min(self.results) if self.results else None,
            "max": max(self.results) if self.results else None,
            "avg": sum(self.results) / len(self.results) if self.results else None
        }

if __name__ == "__main__":
    sample_data = [1, 2, "3", 4.5, "text", 6]
    processor = DataProcessor(sample_data)
    stats = processor.get_stats()
    print(f"Processed {stats['count']} items")
    print(f"Min: {stats['min']}, Max: {stats['max']}, Avg: {stats['avg']}")